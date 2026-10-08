"""
TrafficSentinel AI — Primary Application Server
Real-time vision pipeline orchestrator, web streaming server, and administrative API endpoints.
Pipeline: Video Stream / File -> Vision Inference (YOLOv8 & ByteTrack) -> Infraction Analysis -> ALPR OCR -> Citation Generation -> Multi-Channel Alert Dispatch
"""

import csv
import re
import io
import ipaddress
import cv2
import os
import time
import sqlite3
import threading
import numpy as np
from urllib.parse import urlparse
from datetime import datetime, date, timedelta
from collections import Counter
from flask import (Flask, render_template, Response,
                   jsonify, send_from_directory, send_file,
                   request, session, redirect, url_for, make_response)
from functools import wraps
from ultralytics import YOLO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import easyocr
import config
from citation_builder import build_citation_pdf, compute_infraction_fee, query_prior_offences, BASE_FINES
from alert_dispatcher import dispatch_infraction_alert, dispatch_daily_summary
from registry_lookup import lookup_vehicle_record
from infraction_detector import TrafficInfractionAnalyzer
from plate_scanner import scan_license_plate

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
ADMIN_PASSWORD = config.ADMIN_PASSWORD
DEMO_PASSWORD = config.DEMO_PASSWORD

app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE']   = True
app.config['SESSION_COOKIE_HTTPONLY'] = True

from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)


def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated


def require_admin_api(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('is_admin'):
            return jsonify({"error": "unauthorised"}), 401
        return f(*args, **kwargs)
    return decorated


BASE_DIR        = config.BASE_DIR
DB_PATH         = config.DB_PATH
SCREENSHOT_DIR  = config.SCREENSHOT_DIR
CHALLAN_DIR     = config.CHALLAN_DIR
VIDEO_FOLDER    = config.VIDEO_FOLDER
INPUT_VIDEO_DIR = config.INPUT_VIDEO_DIR
TEST_VIDEO_DIR  = config.TEST_VIDEO_DIR


# ── DATABASE CONTROLLER ────────────────────────────────────
def _get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT,
        video       TEXT,
        violation   TEXT,
        plate       TEXT,
        owner_name  TEXT,
        fine        INTEGER,
        screenshot  TEXT,
        challan     TEXT,
        paid        INTEGER DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS visitors (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp  TEXT,
        ip         TEXT,
        page       TEXT,
        referrer   TEXT,
        ua         TEXT
    )''')
    conn.commit()
    conn.close()


def _log_visitor(page):
    try:
        ip  = request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()
        ref = request.referrer or ''
        ua  = request.user_agent.string[:200] if request.user_agent else ''
        conn = _get_db_connection()
        conn.execute(
            "INSERT INTO visitors (timestamp,ip,page,referrer,ua) VALUES (?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ip, page, ref, ua)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_visitor_stats():
    conn = _get_db_connection()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM visitors")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(DISTINCT ip) FROM visitors")
        unique = c.fetchone()[0]
        c.execute("""SELECT page, COUNT(*) as cnt FROM visitors
                     GROUP BY page ORDER BY cnt DESC""")
        by_page = [{"page": r[0], "count": r[1]} for r in c.fetchall()]
        c.execute("""SELECT DATE(timestamp) as day, COUNT(*) as cnt FROM visitors
                     WHERE DATE(timestamp) >= DATE('now','-6 days')
                     GROUP BY day ORDER BY day""")
        daily = [{"date": r[0], "count": r[1]} for r in c.fetchall()]
        c.execute("""SELECT referrer, COUNT(*) as cnt FROM visitors
                     WHERE referrer != '' GROUP BY referrer
                     ORDER BY cnt DESC LIMIT 10""")
        referrers = [{"referrer": r[0][:80], "count": r[1]} for r in c.fetchall()]
        c.execute("""SELECT timestamp, ip, page, referrer FROM visitors
                     ORDER BY id DESC LIMIT 50""")
        recent = [{"timestamp": r[0], "ip": r[1], "page": r[2],
                   "referrer": r[3][:60]} for r in c.fetchall()]
        return {"total": total, "unique_ips": unique,
                "by_page": by_page, "daily": daily, "referrers": referrers,
                "recent": recent}
    finally:
        conn.close()


def save_violation(video, violation, plate, owner_name, fine, screenshot):
    conn = _get_db_connection()
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO violations (timestamp,video,violation,plate,owner_name,fine,screenshot) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             video, violation, plate, owner_name, fine, screenshot)
        )
        vid = c.lastrowid
        conn.commit()
        return vid
    finally:
        conn.close()


def update_challan(vid, challan_file):
    conn = _get_db_connection()
    try:
        c = conn.cursor()
        c.execute("UPDATE violations SET challan=? WHERE id=?", (challan_file, vid))
        conn.commit()
    finally:
        conn.close()


def get_violations(since_id=0):
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if since_id:
        c.execute("SELECT * FROM violations WHERE id > ? ORDER BY id DESC", (since_id,))
    else:
        c.execute("SELECT * FROM violations ORDER BY id DESC LIMIT 50")
    try:
        rows = [dict(r) for r in c.fetchall()]
        return rows
    finally:
        conn.close()


def get_stats():
    conn = _get_db_connection()
    try:
        c = conn.cursor()
        c.execute(
            "SELECT"
            " COUNT(*) AS total,"
            " SUM(violation LIKE '%NO HELMET%') AS no_helmet,"
            " SUM(violation LIKE '%TRIPLE%') AS triple_riding,"
            " SUM(violation LIKE '%WRONG WAY%') AS wrong_way,"
            " COALESCE(SUM(fine),0) AS total_fines,"
            " SUM(paid=1) AS paid"
            " FROM violations"
        )
        row = c.fetchone()
        total, no_helmet, triple, wrong_way, total_fines, paid = (
            int(row[0] or 0), int(row[1] or 0), int(row[2] or 0),
            int(row[3] or 0), int(row[4] or 0), int(row[5] or 0)
        )
        return {
            "total": total, "no_helmet": no_helmet,
            "triple_riding": triple, "wrong_way": wrong_way,
            "total_fines": total_fines, "paid": paid,
            "pending": total - paid,
            "incentive_pool": int(total_fines * 0.10)
        }
    finally:
        conn.close()


init_db()

# ── INFERENCE MODELS & RECOGNITION ────────────────────────
traffic_model = YOLO(config.TRAFFIC_MODEL_PATH)
helmet_model  = YOLO(config.HELMET_MODEL_PATH)
plate_model   = YOLO(config.PLATE_MODEL_PATH)
reader        = easyocr.Reader(['en'], gpu=False, model_storage_directory=config.EASYOCR_MODEL_DIR)

model_lock    = threading.Lock()
plate_lock    = threading.Lock()


def read_plate(plate_crop):
    return scan_license_plate(plate_crop, reader, plate_lock)


# ── PERFORMANCE METRICS ENGINE ────────────────────────────
class PipelineMetrics:
    LOG_EVERY = 30

    def __init__(self, cam_id):
        self.cam_id         = cam_id
        self.frame_count    = 0
        self.t_last_fps     = time.time()
        self._sum_detection = 0.0
        self._sum_ocr       = 0.0
        self._sum_total     = 0.0
        self._window        = 0
        self.fps            = 0.0
        self.avg_detection  = 0.0
        self.avg_ocr        = 0.0
        self.avg_total      = 0.0

    def record(self, t_detection_ms, t_ocr_ms, t_total_ms):
        self.frame_count += 1
        self._window     += 1
        self._sum_detection += t_detection_ms
        self._sum_ocr       += t_ocr_ms
        self._sum_total     += t_total_ms

        if self._window >= self.LOG_EVERY:
            now      = time.time()
            elapsed  = now - self.t_last_fps
            self.fps = self._window / elapsed if elapsed > 0 else 0

            self.avg_detection = self._sum_detection / self._window
            self.avg_ocr       = self._sum_ocr       / self._window
            self.avg_total     = self._sum_total     / self._window

            print(
                f"  [TrafficSentinel Engine | {self.cam_id}] "
                f"Frame {self.frame_count:>5} | "
                f"Detection: {self.avg_detection:>6.1f}ms | "
                f"OCR: {self.avg_ocr:>5.1f}ms | "
                f"Total: {self.avg_total:>6.1f}ms | "
                f"FPS: {self.fps:.1f}"
            )

            self._sum_detection = 0.0
            self._sum_ocr       = 0.0
            self._sum_total     = 0.0
            self._window        = 0
            self.t_last_fps     = now

    def to_dict(self):
        return {
            "cam_id":           self.cam_id,
            "frame_count":      self.frame_count,
            "fps":              round(self.fps, 1),
            "avg_detection_ms": round(self.avg_detection, 1),
            "avg_ocr_ms":       round(self.avg_ocr, 1),
            "avg_total_ms":     round(self.avg_total, 1),
        }


# ── FRAME PROCESSING PIPELINE ────────────────────────────
PLATE_INTERVAL = 3
VOTE_WINDOW    = 20
MIN_VOTES      = 2
MIN_PLATE_LEN  = 6


def process_frame(frame, state):
    t_frame_start = time.time()
    h_f, w_f = frame.shape[:2]

    t0 = time.time()
    with model_lock:
        traffic_results = traffic_model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        helmet_results  = helmet_model(frame, verbose=False)[0]
    t_detection_ms = (time.time() - t0) * 1000

    traffic_objects  = []
    motorcycle_boxes = []
    traffic_boxes    = []
    helmet_objects   = []

    for box in traffic_results.boxes:
        label = traffic_model.names[int(box.cls)]
        conf  = float(box.conf)
        traffic_objects.append(label)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "motorcycle" and conf >= 0.3:
            motorcycle_boxes.append((x1, y1, x2, y2))
        if label in ("motorcycle", "person") and conf >= 0.3:
            traffic_boxes.append({
                "label": label,
                "id":    int(box.id) if box.id is not None else None,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx":    (x1 + x2) // 2,
                "cy":    (y1 + y2) // 2,
            })

    for box in helmet_results.boxes:
        label = helmet_model.names[int(box.cls)]
        conf  = float(box.conf)
        if label == "nohelmet" and conf < 0.55:
            continue
        helmet_objects.append(label)

    engine     = state["engine"]
    violations = engine.analyze_frame_infractions(traffic_objects, helmet_objects, traffic_boxes, w_f, h_f)

    if "TRIPLE RIDING" in violations and "NO HELMET" not in violations:
        if helmet_objects.count("nohelmet") > 0:
            violations.append("NO HELMET")

    for v in violations:
        state["all_violations_seen"].add(v)
    if "WRONG WAY" in violations:
        state["wrong_way_frames"] = state.get("wrong_way_frames", 0) + 1
        if state["wrong_way_frames"] >= 3:
            state["wrong_way_seen"] = True
    else:
        if not state.get("wrong_way_seen", False):
            state["wrong_way_frames"] = 0

    state["frame_count"] += 1
    if violations:
        state["incident_frame_count"] += 1

    t_ocr_ms = 0.0
    ocr_needed = bool(violations) or state.get("wrong_way_seen", False)
    if ocr_needed and motorcycle_boxes and state["frame_count"] % PLATE_INTERVAL == 0:
        t1 = time.time()
        _run_plate_ocr(frame, motorcycle_boxes, state, h_f, w_f)
        t_ocr_ms = (time.time() - t1) * 1000
    elif not motorcycle_boxes:
        state["cached_plates"] = []

    if not violations:
        if not motorcycle_boxes:
            state["cached_plates"] = []
        if not state.get("wrong_way_seen", False):
            state["plate_history"] = []
            state["last_good_plate"] = ""
        if state["logged"]:
            state["no_violation_frames"] += 1
            if state["no_violation_frames"] >= COOLDOWN_FRAMES:
                state["logged"]               = False
                state["no_violation_frames"]  = 0
                state["incident_frame_count"] = 0
                state["all_violations_seen"]  = set()
                state["wrong_way_seen"]       = False
                state["wrong_way_frames"]     = 0
                state["last_good_plate"]      = ""
                state["engine"].reset()
    else:
        state["no_violation_frames"] = 0

    output_frame = traffic_results.plot()
    display_violations = list(violations)
    if "WRONG WAY" in violations:
        display_violations = [v for v in display_violations if v != "NO HELMET"]
    _draw_annotations(output_frame, display_violations, state["cached_plates"],
                      engine.active_wrong_way_ids, traffic_results, helmet_results,
                      state["label"])

    if _should_log(state):
        state["logged"] = True
        frame_snapshot = output_frame.copy()
        violations_to_log = set(state["all_violations_seen"])
        if "WRONG WAY" in violations_to_log:
            violations_to_log.discard("NO HELMET")
        state_snapshot = {
            "all_violations_seen": violations_to_log,
            "last_good_plate":     state["last_good_plate"],
            "plate_history":       list(state["plate_history"]),
            "label":               state["label"],
            "cam_id":              state["cam_id"],
            "ts":                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        threading.Thread(
            target=_log_violation,
            args=(state_snapshot, frame_snapshot),
            daemon=True
        ).start()

    t_total_ms = (time.time() - t_frame_start) * 1000
    state["metrics"].record(t_detection_ms, t_ocr_ms, t_total_ms)

    return output_frame


def _run_plate_ocr(frame, motorcycle_boxes, state, h_f, w_f):
    motorcycle_boxes = [max(motorcycle_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))]
    state["cached_plates"] = []
    for (mx1, my1, mx2, my2) in motorcycle_boxes:
        pad  = 20
        cx1  = max(0, mx1 - pad); cy1 = max(0, my1 - pad)
        cx2  = min(w_f, mx2 + pad); cy2 = min(h_f, my2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        crop_w = cx2 - cx1; crop_h = cy2 - cy1
        with plate_lock:
            pr = plate_model(crop, verbose=False)[0]
        cands = []
        for pb in pr.boxes:
            cf = float(pb.conf)
            if cf < 0.4:
                continue
            px1, py1, px2, py2 = map(int, pb.xyxy[0])
            pw, ph = px2 - px1, py2 - py1
            if ph == 0 or (pw / ph) < 1.0 or pw > crop_w * 0.95:
                continue
            cands.append((px1, py1, px2, py2, cf))
        if not cands:
            continue
        best_c = max(cands, key=lambda c: c[4])
        px1, py1, px2, py2, _ = best_c
        plate_text = read_plate(crop[py1:py2, px1:px2])
        if plate_text and len(plate_text) >= MIN_PLATE_LEN:
            sc = plate_text[:2]
            _valid = {
                'AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ',
                'HR','HP','JH','JK','KA','KL','LA','LD','MH','ML','MN',
                'MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TR','TS',
                'TG','UK','UP','WB','AN'
            }
            if sc in _valid and re.match(r'^[A-Z]{2}\d{2}', plate_text):
                state["plate_history"].append(plate_text)
                if len(state["plate_history"]) > VOTE_WINDOW:
                    state["plate_history"].pop(0)
        if state["plate_history"]:
            for mc, cnt in Counter(state["plate_history"]).most_common():
                if cnt < MIN_VOTES or len(mc) < MIN_PLATE_LEN:
                    break
                state_code = mc[:2] if len(mc) >= 2 else ""
                valid_states = {
                    'AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ',
                    'HR','HP','JH','JK','KA','KL','LA','LD','MH','ML','MN',
                    'MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TR','TS',
                    'TG','UK','UP','WB','AN'
                }
                if state_code in valid_states:
                    state["last_good_plate"] = mc
                    break
        state["cached_plates"] = [
            (px1 + cx1, py1 + cy1, px2 + cx1, py2 + cy1, state["last_good_plate"])
        ]


def _draw_annotations(frame, violations, cached_plates,
                       wrong_way_ids, traffic_results, helmet_results, label):
    for box in traffic_results.boxes:
        if traffic_model.names[int(box.cls)] != "motorcycle" or box.id is None:
            continue
        if int(box.id) in wrong_way_ids:
            x1, y1, x2, y2 = map(int, box.xyxy[0]); cx = (x1 + x2) // 2
            cv2.arrowedLine(frame, (cx, y1 + 10), (cx, y1 + 50), (0, 0, 255), 3, tipLength=0.4)
            cv2.putText(frame, "WRONG WAY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for box in helmet_results.boxes:
        if helmet_model.names[int(box.cls)] != "licenseplate":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    y = 50
    for v in violations:
        cv2.putText(frame, v, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3); y += 40
    for (px1, py1, px2, py2, pt) in cached_plates:
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(frame, pt if pt else "PLATE", (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, label, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)


def _should_log(state):
    is_wrong_way  = "WRONG WAY" in state["all_violations_seen"]
    has_plate     = bool(state["last_good_plate"])
    n             = state["incident_frame_count"]

    if is_wrong_way:
        return (
            not state["logged"] and
            state["all_violations_seen"] and
            (has_plate or n >= 60)
        )

    return (
        not state["logged"] and
        state["all_violations_seen"] and
        n >= 15 and
        (has_plate or n >= 30)
    )


def _log_violation(state_snapshot, output_frame):
    violation_str = " + ".join(sorted(state_snapshot["all_violations_seen"]))
    if state_snapshot["last_good_plate"]:
        plate_str = state_snapshot["last_good_plate"]
    elif state_snapshot["plate_history"]:
        plate_str = Counter(state_snapshot["plate_history"]).most_common(1)[0][0]
    else:
        plate_str = "UNKNOWN"
    ts    = state_snapshot["ts"]
    label = state_snapshot["label"]

    owner_info  = lookup_vehicle_record(plate_str)
    owner_name  = owner_info["name"]  if owner_info else "Not Available"
    owner_phone = owner_info["phone"] if owner_info else None
    owner_email = owner_info.get("email") if owner_info else None

    violations_list = [v.strip() for v in violation_str.split("+")]

    offence_count    = query_prior_offences(DB_PATH, plate_str) + 1
    _, _, total_fine = compute_infraction_fee(violations_list, offence_count)

    ss_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{state_snapshot['cam_id']}_{plate_str}.jpg"
    cv2.imwrite(os.path.join(SCREENSHOT_DIR, ss_filename), output_frame)

    vid = save_violation(label, violation_str, plate_str,
                         owner_name, total_fine, ss_filename)
    challan_file = build_citation_pdf(
        CHALLAN_DIR, SCREENSHOT_DIR, vid, ts, label,
        violation_str, plate_str, ss_filename, DB_PATH, owner_name,
        offence_count=offence_count
    )
    update_challan(vid, challan_file)
    dispatch_infraction_alert(
        vid, plate_str, violation_str, total_fine,
        ts, os.path.join(CHALLAN_DIR, challan_file),
        owner_name, owner_phone, owner_email
    )


# ── CAMERA MANAGEMENT & STREAMING WORKERS ────────────────
cameras         = {}
cameras_lock    = threading.Lock()
_cam_id_counter = 0

COOLDOWN_FRAMES = 150


def make_camera_state(cam_id, source, label):
    return {
        "cam_id": cam_id, "source": source, "label": label,
        "frame": None, "running": False, "error": None,
        "lock":       threading.Lock(),
        "stop_event": threading.Event(),
        "engine":              TrafficInfractionAnalyzer(),
        "metrics":             PipelineMetrics(cam_id),
        "frame_count":          0,
        "incident_frame_count": 0,
        "no_violation_frames":  0,
        "cached_plates":       [],
        "plate_history":       [],
        "last_good_plate":     "",
        "all_violations_seen": set(),
        "wrong_way_seen":  False,
        "wrong_way_frames": 0,
        "logged":              False,
    }


def get_all_camera_info():
    with cameras_lock:
        return [{"cam_id": s["cam_id"], "source": s["source"],
                 "label": s["label"], "running": s["running"],
                 "error": s["error"]} for s in cameras.values()]


def process_source(cam_id):
    with cameras_lock:
        if cam_id not in cameras:
            return
        state = cameras[cam_id]

    source  = state["source"]
    is_rtsp = any(source.startswith(p) for p in ("rtsp://", "rtmp://", "http"))
    cap     = cv2.VideoCapture(source)

    if not cap.isOpened():
        with state["lock"]:
            state["running"] = False
            state["error"]   = f"Cannot open: {source}"
        return

    with state["lock"]:
        state["running"] = True
        state["error"]   = None

    if is_rtsp:
        process_every = 1
    else:
        video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        process_every = max(1, round(video_fps / 5))

    raw_idx = 0

    while not state["stop_event"].is_set():
        ret, frame = cap.read()
        if not ret:
            if is_rtsp:
                cap.release(); time.sleep(2)
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    with state["lock"]:
                        state["error"] = "Stream disconnected"
                    break
                continue
            else:
                with state["lock"]:
                    state["running"] = False
                    done_frame = _make_text_frame("Video playback complete.", os.path.basename(source), (0, 229, 255))
                    state["frame"]   = done_frame
                break

        raw_idx += 1
        if raw_idx % process_every != 0:
            continue

        try:
            output_frame = process_frame(frame, state)
        except Exception as exc:
            print(f"  [TrafficSentinel Stream | {cam_id}] Frame processing error: {exc}")
            continue

        ret2, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret2:
            with state["lock"]:
                state["frame"] = buffer.tobytes()

    cap.release()
    with state["lock"]:
        state["running"] = False


def _make_text_frame(line1, line2="", color=(0, 229, 255)):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, line1, (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if line2:
        cv2.putText(img, line2, (40, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

_PLACEHOLDER_FRAME = _make_text_frame("Initializing TrafficSentinel AI...", "Loading Vision AI Engine")

def _placeholder_frame():
    return _PLACEHOLDER_FRAME


def gen_frames_for(cam_id):
    loading = _placeholder_frame()
    while True:
        with cameras_lock:
            state = cameras.get(cam_id)
        if state:
            with state["lock"]:
                frame = state["frame"]
                error = state.get("error")
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            elif error:
                if state.get('_cached_err_msg') != error:
                    state['_cached_err_frame'] = _make_text_frame('Stream Error:', error[:55], (0, 80, 255))
                    state['_cached_err_msg']   = error
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + state['_cached_err_frame'] + b'\r\n')
            else:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + loading + b'\r\n')
        else:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + loading + b'\r\n')
        time.sleep(0.04)


# ── STORAGE UTILITIES ─────────────────────────────────────
def get_available_videos():
    video_set = set()
    for dir_path in [VIDEO_FOLDER, INPUT_VIDEO_DIR, TEST_VIDEO_DIR]:
        if os.path.exists(dir_path):
            for f in os.listdir(dir_path):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_set.add(f)
    return sorted(list(video_set))


def resolve_video_path(video_name):
    safe_name = os.path.basename(video_name)
    for dir_path in [INPUT_VIDEO_DIR, TEST_VIDEO_DIR, VIDEO_FOLDER]:
        target = os.path.join(dir_path, safe_name)
        if os.path.isfile(target):
            return target
    return None


# ── HTTP ROUTE CONTROLLERS ─────────────────────────────────
_login_attempts      = {}
_login_attempts_lock = threading.Lock()
_MAX_ATTEMPTS        = 10
_LOCKOUT_SECS        = 900
_MAX_IPS             = 10_000


def _is_rate_limited(ip):
    now = time.time()
    with _login_attempts_lock:
        times = [t for t in _login_attempts.get(ip, []) if now - t < _LOCKOUT_SECS]
        _login_attempts[ip] = times
        return len(times) >= _MAX_ATTEMPTS


def _record_attempt(ip):
    with _login_attempts_lock:
        if len(_login_attempts) >= _MAX_IPS:
            oldest = min(_login_attempts, key=lambda k: _login_attempts[k][-1] if _login_attempts[k] else 0)
            del _login_attempts[oldest]
        _login_attempts.setdefault(ip, []).append(time.time())


@app.route('/login', methods=['GET', 'POST'])
def login():
    _log_visitor('/login')
    error = None
    ip    = request.remote_addr
    if request.method == 'POST':
        submitted = request.form.get('password', '')
        is_admin  = (submitted == ADMIN_PASSWORD)
        is_demo   = bool(DEMO_PASSWORD) and (submitted == DEMO_PASSWORD)
        if _is_rate_limited(ip):
            error = 'Too many failed login attempts. Account temporarily locked for 15 minutes.'
        elif is_admin or is_demo:
            session['is_admin'] = True
            session['is_demo']  = is_demo and not is_admin
            _login_attempts.pop(ip, None)
            next_url  = request.args.get('next', '/')
            parsed    = urlparse(next_url)
            safe_next = next_url if (not parsed.scheme and not parsed.netloc) else '/'
            return redirect(safe_next)
        else:
            _record_attempt(ip)
            error = 'Invalid authentication password'
    return render_template(
        'login.html', error=error,
        demo_enabled=bool(DEMO_PASSWORD),
        demo_password=DEMO_PASSWORD if DEMO_PASSWORD else ''
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/citizen')


@app.route('/')
@require_admin
def index():
    _log_visitor('/dashboard')
    videos = get_available_videos()
    return render_template('index.html', videos=videos,
                           is_demo=session.get('is_demo', False))


@app.route('/analytics')
@require_admin
def analytics():
    _log_visitor('/analytics')
    return render_template('analytics.html')


@app.route('/cameras')
@require_admin_api
def list_cameras():
    return jsonify(get_all_camera_info())


@app.route('/upload', methods=['POST'])
@require_admin_api
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov', '.mkv']:
        return jsonify({"error": "Unsupported video format. Allowed: MP4, AVI, MOV, MKV"}), 400

    target_path = os.path.join(INPUT_VIDEO_DIR, filename)
    file.save(target_path)
    return jsonify({"status": "success", "filename": filename, "path": target_path})


@app.route('/camera/add', methods=['POST'])
@require_admin_api
def add_camera():
    data   = request.get_json()
    source = data.get("source", "").strip()
    label  = data.get("label", "Camera").strip()
    if not source:
        return jsonify({"error": "Video source parameters required"}), 400

    if any(source.startswith(p) for p in ("rtsp://", "rtmp://", "http")):
        parsed_host = urlparse(source).hostname or ""
        try:
            addr = ipaddress.ip_address(parsed_host)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                return jsonify({"error": "Private IP addresses are restricted"}), 400
        except ValueError:
            import socket
            try:
                resolved = socket.getaddrinfo(parsed_host, None)
                for r in resolved:
                    try:
                        raddr = ipaddress.ip_address(r[4][0])
                        if raddr.is_private or raddr.is_loopback or raddr.is_link_local:
                            return jsonify({"error": "Private IP networks restricted"}), 400
                    except ValueError:
                        pass
            except socket.gaierror:
                return jsonify({"error": "Could not resolve stream hostname"}), 400
    else:
        resolved = resolve_video_path(source)
        if resolved:
            source = resolved
        else:
            safe_name = os.path.basename(source)
            source    = os.path.join(INPUT_VIDEO_DIR, safe_name)

    global _cam_id_counter
    with cameras_lock:
        _cam_id_counter += 1
        cam_id = f"cam_{_cam_id_counter}"
        cameras[cam_id] = make_camera_state(cam_id, source, label)
    threading.Thread(target=process_source, args=(cam_id,), daemon=True).start()
    return jsonify({"status": "started", "cam_id": cam_id, "label": label})


@app.route('/camera/stop/<cam_id>')
@require_admin_api
def stop_camera(cam_id):
    with cameras_lock:
        state = cameras.get(cam_id)
    if state:
        state["stop_event"].set()
        with state["lock"]:
            state["running"] = False
    return jsonify({"status": "stopped", "cam_id": cam_id})


@app.route('/camera/remove/<cam_id>')
@require_admin_api
def remove_camera(cam_id):
    with cameras_lock:
        state = cameras.pop(cam_id, None)
    if state:
        state["stop_event"].set()
        with state["lock"]:
            state["running"] = False
    return jsonify({"status": "removed", "cam_id": cam_id})


@app.route('/camera/stop_all')
@require_admin_api
def stop_all_cameras():
    with cameras_lock:
        states = list(cameras.values())
    for state in states:
        state["stop_event"].set()
        with state["lock"]:
            state["running"] = False
    return jsonify({"status": "all streams stopped"})


@app.route('/video_feed/<cam_id>')
def video_feed(cam_id):
    if not session.get('is_admin'):
        return Response(status=401)
    return Response(gen_frames_for(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed_legacy():
    if not session.get('is_admin'):
        return Response(status=401)
    def dynamic_gen():
        placeholder = _placeholder_frame()
        while True:
            with cameras_lock:
                cam_id = next(iter(cameras), None)
            if cam_id:
                with cameras_lock:
                    state = cameras.get(cam_id)
                if state:
                    with state["lock"]:
                        frame = state["frame"]
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + (frame if frame else placeholder) + b'\r\n')
                    time.sleep(0.04)
                    continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + placeholder + b'\r\n')
            time.sleep(0.1)
    return Response(dynamic_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start/<video_name>')
@require_admin_api
def start_video(video_name):
    source = resolve_video_path(video_name)
    if not source or not os.path.isfile(source):
        return jsonify({"error": "Video file not found"}), 404

    safe_name = os.path.basename(source)
    with cameras_lock:
        old_states = list(cameras.values())
    for s in old_states:
        s["stop_event"].set()
        with s["lock"]:
            s["running"] = False
    deadline = time.time() + 1.0
    for s in old_states:
        remaining = max(0.0, deadline - time.time())
        s["stop_event"].wait(timeout=remaining)
    with cameras_lock:
        cameras.clear()
    global _cam_id_counter
    with cameras_lock:
        _cam_id_counter += 1
        cam_id = f"cam_{_cam_id_counter}"
        cameras[cam_id] = make_camera_state(cam_id, source, safe_name)
    threading.Thread(target=process_source, args=(cam_id,), daemon=True).start()
    return jsonify({"status": "started", "video": safe_name, "cam_id": cam_id})


@app.route('/stop')
@require_admin_api
def stop_video():
    with cameras_lock:
        states = list(cameras.values())
    for state in states:
        state["stop_event"].set()
        with state["lock"]:
            state["running"] = False
    return jsonify({"status": "stopped"})


@app.route('/status')
@require_admin_api
def status_api():
    with cameras_lock:
        running = any(s["running"] for s in cameras.values())
        vids    = [s["label"] for s in cameras.values() if s["running"]]
    return jsonify({"running": running, "video": vids[0] if vids else ""})


@app.route('/metrics')
@require_admin_api
def metrics_api():
    with cameras_lock:
        return jsonify([s["metrics"].to_dict() for s in cameras.values()])


@app.route('/violations')
@require_admin_api
def violations_api():
    since = request.args.get('since', 0, type=int)
    return jsonify(get_violations(since_id=since))


@app.route('/stats')
def stats_api():
    return jsonify(get_stats())


@app.route('/challan/<int:vid>')
@require_admin
def download_challan(vid):
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM violations WHERE id=?", (vid,))
    row = c.fetchone()
    if not row:
        conn.close()
        return "Citation Record Not Found", 404
    row = dict(row)
    if not row.get('challan'):
        cf = build_citation_pdf(
            CHALLAN_DIR, SCREENSHOT_DIR,
            row['id'], row['timestamp'], row['video'],
            row['violation'], row['plate'],
            row['screenshot'] or "", DB_PATH,
            row.get('owner_name', 'Not Available')
        )
        update_challan(vid, cf)
    else:
        cf = row['challan']
    return send_file(os.path.join(CHALLAN_DIR, cf),
                     as_attachment=True, download_name=cf)


@app.route('/daily_summary')
@require_admin_api
def daily_summary():
    stats = get_stats()
    threading.Thread(target=dispatch_daily_summary, args=(stats,), daemon=True).start()
    return jsonify({"status": "dispatching", "stats": stats})


@app.route('/static/screenshots/<filename>')
def screenshot(filename):
    if not session.get('is_admin'):
        return Response(status=401)
    safe = os.path.basename(filename)
    return send_from_directory(SCREENSHOT_DIR, safe)


@app.route('/analytics_data')
@require_admin_api
def analytics_data():
    conn = _get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM violations WHERE violation LIKE '%NO HELMET%'")
    no_helmet = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM violations WHERE violation LIKE '%TRIPLE%'")
    triple = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM violations WHERE violation LIKE '%WRONG WAY%'")
    wrong_way = c.fetchone()[0]
    by_type = {}
    if no_helmet:
        by_type["No Helmet"] = no_helmet
    if triple:
        by_type["Triple Riding"] = triple
    if wrong_way:
        by_type["Wrong Way"] = wrong_way
    c.execute("""SELECT DATE(timestamp) as day, COUNT(*) as cnt FROM violations
                 WHERE DATE(timestamp) >= DATE('now','-6 days') GROUP BY day ORDER BY day""")
    day_map = {r[0]: r[1] for r in c.fetchall()}
    today   = date.today()
    daily   = []
    for i in range(6, -1, -1):
        d = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        daily.append({"date": d[5:], "count": day_map.get(d, 0)})
    c.execute("""SELECT DATE(timestamp) as day, COALESCE(SUM(fine),0) as total
                 FROM violations WHERE DATE(timestamp) >= DATE('now','-6 days')
                 GROUP BY day ORDER BY day""")
    fine_map    = {r[0]: r[1] for r in c.fetchall()}
    daily_fines = []
    for i in range(6, -1, -1):
        d = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        daily_fines.append({"date": d[5:], "total": fine_map.get(d, 0)})
    c.execute("""SELECT CAST(strftime('%H',timestamp) AS INTEGER) as hr, COUNT(*) as cnt
                 FROM violations GROUP BY hr ORDER BY hr""")
    hourly = [{"hour": r[0], "count": r[1]} for r in c.fetchall()]
    c.execute("""SELECT plate, owner_name, COUNT(*) as cnt,
                        GROUP_CONCAT(DISTINCT violation) as all_violations,
                        COALESCE(SUM(fine),0) as total_fine
                 FROM violations WHERE plate != 'UNKNOWN'
                 GROUP BY plate ORDER BY cnt DESC, total_fine DESC LIMIT 10""")
    top_plates = [{"plate": r[0], "owner": r[1], "count": r[2],
                   "violations": r[3] or "", "total_fine": r[4]} for r in c.fetchall()]
    c.execute("SELECT COALESCE(SUM(fine),0) FROM violations WHERE paid=1")
    paid_fines = c.fetchone()[0]
    conn.close()
    return jsonify({"by_type": by_type, "daily": daily, "daily_fines": daily_fines,
                    "hourly": hourly, "top_plates": top_plates, "paid_fines": paid_fines})


@app.route('/violation/<int:vid>/paid', methods=['PATCH'])
@require_admin_api
def mark_paid(vid):
    conn = _get_db_connection()
    c = conn.cursor()
    c.execute("SELECT paid FROM violations WHERE id=?", (vid,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Infraction record not found"}), 404
    new_status = 0 if row[0] == 1 else 1
    c.execute("UPDATE violations SET paid=? WHERE id=?", (new_status, vid))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "paid": new_status})


@app.route('/export')
@require_admin
def export_csv():
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id,timestamp,video,violation,plate,owner_name,fine,paid FROM violations ORDER BY id DESC").fetchall()
    conn.close()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Citation No", "Timestamp", "Camera Stream", "Infraction", "Plate Number", "Vehicle Owner", "Fine (Rs.)", "Status"])
    for r in rows:
        w.writerow([f"TS-{r['id']:06d}", r['timestamp'], r['video'],
                    r['violation'], r['plate'], r['owner_name'],
                    r['fine'], "Paid" if r['paid'] else "Pending"])
    out.seek(0)
    resp = make_response(out.getvalue())
    resp.headers["Content-Disposition"] = "attachment; filename=traffic_sentinel_infractions.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


@app.route('/citizen')
def citizen_portal():
    _log_visitor('/citizen')
    return render_template('citizen.html')


@app.route('/citizen/violations')
def citizen_violations():
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id,timestamp,video,violation,plate,fine,paid,screenshot FROM violations ORDER BY id DESC LIMIT 50"
    ).fetchall()
    conn.close()
    def _mask_plate(p):
        if not p or p == "UNKNOWN":
            return "UNKNOWN"
        return p[:2] + "*" * max(0, len(p) - 4) + p[-2:] if len(p) > 4 else "****"
    return jsonify([{"id": r["id"], "challan": f"TS-{r['id']:06d}", "timestamp": r["timestamp"],
                     "camera": r["video"], "violation": r["violation"],
                     "plate": _mask_plate(r["plate"]),
                     "fine": r["fine"], "paid": bool(r["paid"]),
                     "screenshot": None} for r in rows])


@app.route('/citizen/stats')
def citizen_stats():
    conn = _get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM violations")
    total = c.fetchone()[0]
    c.execute("SELECT COALESCE(SUM(fine),0) FROM violations")
    total_fines = c.fetchone()[0]
    c.execute("SELECT COALESCE(SUM(fine),0) FROM violations WHERE paid=1")
    collected = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM violations WHERE violation LIKE '%NO HELMET%'")
    nh = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM violations WHERE violation LIKE '%TRIPLE%'")
    tr = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM violations WHERE violation LIKE '%WRONG WAY%'")
    ww = c.fetchone()[0]
    conn.close()
    return jsonify({"total_violations": total, "total_fines": total_fines,
                    "fines_collected": collected, "incentive_pool": int(collected * 0.10),
                    "no_helmet": nh, "triple_riding": tr, "wrong_way": ww})


@app.route('/visitors')
@require_admin
def visitor_stats():
    stats = get_visitor_stats()
    if session.get('is_demo'):
        stats.pop('recent', None)
    return jsonify(stats)


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port  = config.PORT
    host  = config.HOST
    print(f"🚀 Starting TrafficSentinel AI Server on http://{host}:{port}")
    app.run(debug=debug, threaded=True, host=host, port=port)

"""
TrafficSentinel AI — Batch Video Processor Engine
Processes video feeds/files, tracks traffic entities, analyzes infractions, performs ALPR OCR, and generates annotated output clips.
"""

import cv2
import os
import re
import numpy as np
import easyocr
import threading
from collections import Counter, defaultdict
from ultralytics import YOLO

import config
from plate_scanner import scan_license_plate

# Vision Inference Models Initialization
traffic_detector = YOLO(config.TRAFFIC_MODEL_PATH)
helmet_detector  = YOLO(config.HELMET_MODEL_PATH)
plate_detector   = YOLO(config.PLATE_MODEL_PATH)

_ocr_engine = easyocr.Reader(['en'], gpu=False, model_storage_directory=config.EASYOCR_MODEL_DIR)
_ocr_lock   = threading.Lock()

WRONG_WAY_WINDOW_FRAMES = 5
WRONG_WAY_MIN_DELTA_Y   = -5

PLATE_SEARCH_INTERVAL = 5
PLATE_VOTE_WINDOW     = 20
PLATE_MIN_VOTES       = 2


def read_plate(crop_img):
    return scan_license_plate(crop_img, _ocr_engine, _ocr_lock)


def compute_bounding_overlap(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    return intersection / area_a


def initialize_video_writer(output_file_path, frame_w, frame_h, fps_rate):
    fourcc_code = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_file_path, fourcc_code, fps_rate, (frame_w, frame_h))
    if video_writer.isOpened():
        print(f"  Codec initialized: avc1 -> {output_file_path}")
        return video_writer, output_file_path

    fourcc_code = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file_path, fourcc_code, fps_rate, (frame_w, frame_h))
    if video_writer.isOpened():
        print(f"  Codec initialized: mp4v -> {output_file_path}")
        return video_writer, output_file_path

    fallback_avi_path = os.path.splitext(output_file_path)[0] + '.avi'
    fourcc_code = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(fallback_avi_path, fourcc_code, fps_rate, (frame_w, frame_h))
    print(f"  Codec fallback initialized: XVID -> {fallback_avi_path}")
    return video_writer, fallback_avi_path


def process_all_input_videos():
    output_dir = os.path.join(config.OUTPUT_DIR, "video_results")
    os.makedirs(output_dir, exist_ok=True)

    video_sources = []
    for dir_path in [config.INPUT_VIDEO_DIR, config.TEST_VIDEO_DIR, config.VIDEO_FOLDER]:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_sources.append(os.path.join(dir_path, filename))

    if not video_sources:
        print("No input video files located in storage directories.")
        return

    print(f"TrafficSentinel AI — Processing {len(video_sources)} video file(s)...")

    for src_path in video_sources:
        src_name = os.path.basename(src_path)
        dest_filename = f"processed_{src_name}"
        dest_path = os.path.join(output_dir, os.path.splitext(dest_filename)[0] + ".mp4")

        print(f"\nProcessing Video: {src_name}")

        capture = cv2.VideoCapture(src_path)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS)) or 25
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Specs: {w}x{h} | FPS: {fps} | Total Frames: {total_frames}")

        writer, actual_dest_path = initialize_video_writer(dest_path, w, h, fps)
        if not writer.isOpened():
            print("  ❌ Unable to initialize VideoWriter. Skipping file.")
            capture.release()
            continue

        frame_counter = 0
        cached_plate_overlays = []
        historical_plates = []
        last_recognized_plate = ""
        track_cy_log = defaultdict(list)
        counterflow_ids = set()
        recorded_infractions = []

        while True:
            success, frame_data = capture.read()
            if not success:
                break
            frame_counter += 1

            if frame_counter % 30 == 0:
                print(f"  Analyzing frame {frame_counter}/{total_frames}...", end='\r')

            t_results = traffic_detector.track(
                frame_data, persist=True, tracker="bytetrack.yaml", verbose=False
            )[0]
            h_results = helmet_detector(frame_data, verbose=False)[0]

            traffic_labels = []
            motorcycle_boxes = []
            headwear_labels = []
            person_boxes = []

            for box in t_results.boxes:
                lbl = traffic_detector.names[int(box.cls)]
                conf = float(box.conf)
                traffic_labels.append(lbl)
                if lbl == "motorcycle" and conf >= 0.3:
                    motorcycle_boxes.append(list(map(int, box.xyxy[0])))
                if lbl == "person" and conf >= 0.4:
                    person_boxes.append(list(map(int, box.xyxy[0])))

            for box in h_results.boxes:
                headwear_labels.append(helmet_detector.names[int(box.cls)])

            # Counterflow / Wrong Way detection logic
            for box in t_results.boxes:
                if traffic_detector.names[int(box.cls)] != "motorcycle" or box.id is None:
                    continue
                track_id = int(box.id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cx < w * 0.15 or cx > w * 0.85:
                    counterflow_ids.discard(track_id)
                    continue

                b_area = (x2 - x1) * (y2 - y1)
                if b_area < (w * h * 0.01):
                    counterflow_ids.discard(track_id)
                    continue

                track_cy_log[track_id].append(cy)
                if len(track_cy_log[track_id]) > WRONG_WAY_WINDOW_FRAMES + 2:
                    track_cy_log[track_id].pop(0)

                if len(track_cy_log[track_id]) >= WRONG_WAY_WINDOW_FRAMES:
                    cy_history = track_cy_log[track_id]
                    delta_y = (cy_history[-1] - cy_history[-WRONG_WAY_WINDOW_FRAMES]) / WRONG_WAY_WINDOW_FRAMES
                    if delta_y < WRONG_WAY_MIN_DELTA_Y:
                        counterflow_ids.add(track_id)
                    else:
                        counterflow_ids.discard(track_id)

            # Triple Riding check
            triple_riding_active = False
            for (mx1, my1, mx2, my2) in motorcycle_boxes:
                bw, bh = mx2 - mx1, my2 - my1
                bike_bounds = (
                    max(0, mx1 - int(bw * 0.05)), max(0, my1 - int(bh * 0.05)),
                    min(w, mx2 + int(bw * 0.05)), min(h, my2 + int(bh * 0.05))
                )
                r_count = sum(
                    1 for (px1, py1, px2, py2) in person_boxes
                    if compute_bounding_overlap((px1, py1, px2, py2), bike_bounds) > 0.50
                )
                if r_count >= 3:
                    triple_riding_active = True
                    break

            unhelmeted_cnt = headwear_labels.count("nohelmet")
            helmeted_cnt = headwear_labels.count("helmet") + headwear_labels.count("motorcyclist")
            no_helmet_active = ("motorcycle" in traffic_labels and unhelmeted_cnt > 0 and unhelmeted_cnt > helmeted_cnt)

            active_infractions = []
            if no_helmet_active:
                active_infractions.append("NO HELMET")
            if triple_riding_active:
                active_infractions.append("TRIPLE RIDING")
            if counterflow_ids:
                active_infractions.append("WRONG WAY")

            if "WRONG WAY" in active_infractions and "NO HELMET" in active_infractions:
                active_infractions.remove("NO HELMET")

            for item in active_infractions:
                if item not in recorded_infractions:
                    recorded_infractions.append(item)

            if active_infractions and motorcycle_boxes and frame_counter % PLATE_SEARCH_INTERVAL == 0:
                mx1, my1, mx2, my2 = max(motorcycle_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                pad_x = int((mx2 - mx1) * 0.2)
                pad_y_top = int((my2 - my1) * 0.05)
                pad_y_bot = int((my2 - my1) * 0.35)
                cx1 = max(0, mx1 - pad_x)
                cy1 = max(0, my1 - pad_y_top)
                cx2 = min(w, mx2 + pad_x)
                cy2 = min(h, my2 + pad_y_bot)
                crop = frame_data[cy1:cy2, cx1:cx2]

                if crop.size > 0:
                    crop_w, crop_h = cx2 - cx1, cy2 - cy1
                    p_res = plate_detector(crop, verbose=False)[0]
                    cands = []
                    for pb in p_res.boxes:
                        conf = float(pb.conf)
                        if conf < 0.40:
                            continue
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        pw, ph = px2 - px1, py2 - py1
                        if ph == 0 or (pw / ph) < 1.0 or pw > crop_w * 0.95:
                            continue
                        cands.append((px1, py1, px2, py2, conf))

                    if cands:
                        px1, py1, px2, py2, _ = max(cands, key=lambda c: c[4])
                        scanned_text = read_plate(crop[py1:py2, px1:px2])
                        if scanned_text:
                            historical_plates.append(scanned_text)
                            if len(historical_plates) > PLATE_VOTE_WINDOW:
                                historical_plates.pop(0)

                        if historical_plates:
                            most_common_text, match_count = Counter(historical_plates).most_common(1)[0]
                            if match_count >= PLATE_MIN_VOTES and len(most_common_text) >= 8:
                                last_recognized_plate = most_common_text
                        cached_plate_overlays = [(px1 + cx1, py1 + cy1, px2 + cx1, py2 + cy1, last_recognized_plate)]

            if not active_infractions:
                cached_plate_overlays = []
                historical_plates = []

            annotated_frame = t_results.plot()

            for box in t_results.boxes:
                if traffic_detector.names[int(box.cls)] != "motorcycle" or box.id is None:
                    continue
                if int(box.id) in counterflow_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cv2.arrowedLine(annotated_frame, (cx, y1 + 10), (cx, y1 + 50), (0, 0, 255), 3, tipLength=0.4)
                    cv2.putText(annotated_frame, "WRONG WAY", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            for box in h_results.boxes:
                if helmet_detector.names[int(box.cls)] != "licenseplate":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            y_pos = 50
            for item in active_infractions:
                cv2.putText(annotated_frame, item, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                y_pos += 40

            for (px1, py1, px2, py2, plate_text) in cached_plate_overlays:
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(
                    annotated_frame, plate_text if plate_text else "PLATE",
                    (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

            writer.write(annotated_frame)

        capture.release()
        writer.release()

        if os.path.exists(actual_dest_path) and os.path.getsize(actual_dest_path) > 0:
            mb_size = os.path.getsize(actual_dest_path) / (1024 * 1024)
            print(f"\n  ✅ Saved annotated output -> {actual_dest_path} ({mb_size:.1f} MB)")
            if recorded_infractions:
                print(f"  Infractions recorded: {', '.join(recorded_infractions)}")
            else:
                print("  No infractions recorded.")
        else:
            print(f"\n  ❌ Failed to write video file: {actual_dest_path}")

    print("\nBatch video processing complete. Files saved in 'outputs/video_results'.")


if __name__ == "__main__":
    process_all_input_videos()

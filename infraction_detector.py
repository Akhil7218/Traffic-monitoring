"""
TrafficSentinel AI — Real-Time Traffic Infraction Detection Engine
Analyzes video bounding boxes, tracking trajectories, and object overlaps to flag safety violations.
"""

from collections import defaultdict, deque


class TrafficInfractionAnalyzer:
    WRONG_WAY_WINDOW_FRAMES = 5
    WRONG_WAY_MIN_DELTA_Y = 5
    MARGIN_BOUND_RATIO = 0.15

    def __init__(self):
        self.cy_history_tracker = defaultdict(lambda: deque(maxlen=self.WRONG_WAY_WINDOW_FRAMES + 2))
        self.area_history_tracker = defaultdict(lambda: deque(maxlen=self.WRONG_WAY_WINDOW_FRAMES + 2))
        self.active_wrong_way_ids = set()

    def analyze_frame_infractions(self, detected_traffic_labels, detected_headwear_labels, traffic_bboxes, frame_width, frame_height=0):
        """
        Main evaluation method for frame-level traffic violations.
        """
        self._update_counterflow_tracks(traffic_bboxes, frame_width)
        infractions = []

        if self.check_no_helmet(detected_traffic_labels, detected_headwear_labels):
            infractions.append("NO HELMET")

        person_boxes = [
            (b["x1"], b["y1"], b["x2"], b["y2"])
            for b in traffic_bboxes if b.get("label") == "person"
        ]
        motorcycle_boxes = [
            (b["x1"], b["y1"], b["x2"], b["y2"])
            for b in traffic_bboxes if b.get("label") == "motorcycle"
        ]

        if self.check_triple_riding(detected_traffic_labels, person_boxes, motorcycle_boxes, frame_width, frame_height):
            infractions.append("TRIPLE RIDING")

        if self.check_wrong_way():
            infractions.append("WRONG WAY")

        return infractions

    def check_no_helmet(self, traffic_labels, headwear_labels):
        if "motorcycle" not in traffic_labels:
            return False
        unhelmeted_riders = headwear_labels.count("nohelmet")
        helmeted_riders = headwear_labels.count("helmet")
        return unhelmeted_riders > 0 and unhelmeted_riders > helmeted_riders

    def check_triple_riding(self, traffic_labels, person_boxes=None, motorcycle_boxes=None, frame_w=0, frame_h=0):
        if "motorcycle" not in traffic_labels:
            return False
        if not person_boxes or not motorcycle_boxes:
            return traffic_labels.count("person") >= 3

        for (mx1, my1, mx2, my2) in motorcycle_boxes:
            mw, mh = mx2 - mx1, my2 - my1
            expanded_x1 = max(0, mx1 - int(mw * 0.10))
            expanded_y1 = max(0, my1 - int(mh * 0.80))
            expanded_x2 = min(frame_w if frame_w else mx2 + mw, mx2 + int(mw * 0.10))
            expanded_y2 = min(frame_h if frame_h else my2 + mh, my2 + int(mh * 0.05))

            rider_count = sum(
                1 for (px1, py1, px2, py2) in person_boxes
                if self._compute_overlap_ratio((px1, py1, px2, py2), (expanded_x1, expanded_y1, expanded_x2, expanded_y2)) > 0.30
            )
            if rider_count >= 3:
                return True
        return False

    @staticmethod
    def _compute_overlap_ratio(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box_a_area = max((ax2 - ax1) * (ay2 - ay1), 1)
        return inter_area / box_a_area

    def check_wrong_way(self):
        return len(self.active_wrong_way_ids) > 0

    def reset(self):
        self.cy_history_tracker.clear()
        self.area_history_tracker.clear()
        self.active_wrong_way_ids.clear()

    # ── INTERNAL TRAJECTORY MONITOR ──────────────────────────

    def _update_counterflow_tracks(self, traffic_bboxes, frame_width):
        for b in traffic_bboxes:
            if b.get("label") != "motorcycle" or b.get("id") is None:
                continue

            track_id = b["id"]
            cx = b["cx"]
            cy = b["cy"]
            box_area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

            if cx < frame_width * self.MARGIN_BOUND_RATIO or cx > frame_width * (1.0 - self.MARGIN_BOUND_RATIO):
                self.active_wrong_way_ids.discard(track_id)
                continue

            cy_hist = self.cy_history_tracker[track_id]
            area_hist = self.area_history_tracker[track_id]
            cy_hist.append(cy)
            area_hist.append(box_area)

            if len(cy_hist) < self.WRONG_WAY_WINDOW_FRAMES:
                continue

            delta_y = (cy_hist[-1] - cy_hist[-self.WRONG_WAY_WINDOW_FRAMES]) / self.WRONG_WAY_WINDOW_FRAMES
            is_moving_upward = delta_y < -self.WRONG_WAY_MIN_DELTA_Y

            initial_area = area_hist[-self.WRONG_WAY_WINDOW_FRAMES]
            if initial_area > 0:
                growth_rate = (area_hist[-1] - initial_area) / (initial_area * self.WRONG_WAY_WINDOW_FRAMES)
                is_approaching_rapidly = growth_rate > 0.25
            else:
                is_approaching_rapidly = False

            if is_moving_upward or is_approaching_rapidly:
                self.active_wrong_way_ids.add(track_id)
            else:
                self.active_wrong_way_ids.discard(track_id)


# Backward Compatibility Alias
ViolationEngine = TrafficInfractionAnalyzer

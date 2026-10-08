"""
TrafficSentinel AI — Automated Test Suite
Unit tests for Infraction Detector, Citation Builder, and Registration Pattern Matching.
Run with: python -m pytest tests/ -v
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infraction_detector import TrafficInfractionAnalyzer
from citation_builder import compute_infraction_fee
from plate_scanner import _REGISTRATION_PATTERNS


# ══════════════════════════════════════════════════════════
# Infraction Detector — Helmet Compliance Tests
# ══════════════════════════════════════════════════════════

class TestNoHelmet:

    def setup_method(self):
        self.engine = TrafficInfractionAnalyzer()

    def test_flags_when_nohelmet_outnumbers_helmet(self):
        assert self.engine.check_no_helmet(
            traffic_labels=["motorcycle"],
            headwear_labels=["nohelmet", "nohelmet"]
        ) is True

    def test_clears_when_helmet_outnumbers_nohelmet(self):
        assert self.engine.check_no_helmet(
            traffic_labels=["motorcycle"],
            headwear_labels=["helmet", "helmet", "nohelmet"]
        ) is False

    def test_no_flag_without_motorcycle(self):
        assert self.engine.check_no_helmet(
            traffic_labels=["car", "bus"],
            headwear_labels=["nohelmet"]
        ) is False

    def test_no_flag_when_no_helmet_objects(self):
        assert self.engine.check_no_helmet(
            traffic_labels=["motorcycle"],
            headwear_labels=[]
        ) is False

    def test_motorcyclist_label_is_neutral(self):
        assert self.engine.check_no_helmet(
            traffic_labels=["motorcycle"],
            headwear_labels=["motorcyclist", "nohelmet"]
        ) is True

    def test_confirmed_helmet_suppresses_flag(self):
        assert self.engine.check_no_helmet(
            traffic_labels=["motorcycle"],
            headwear_labels=["helmet", "nohelmet"]
        ) is False


# ══════════════════════════════════════════════════════════
# Infraction Detector — Triple Riding Tests
# ══════════════════════════════════════════════════════════

class TestTripleRiding:

    def setup_method(self):
        self.engine = TrafficInfractionAnalyzer()

    def test_flags_three_persons(self):
        assert self.engine.check_triple_riding(
            ["motorcycle", "person", "person", "person"]
        ) is True

    def test_flags_more_than_three(self):
        assert self.engine.check_triple_riding(
            ["motorcycle", "person", "person", "person", "person"]
        ) is True

    def test_no_flag_with_two_persons(self):
        assert self.engine.check_triple_riding(
            ["motorcycle", "person", "person"]
        ) is False

    def test_no_flag_without_motorcycle(self):
        assert self.engine.check_triple_riding(
            ["car", "person", "person", "person"]
        ) is False

    def test_no_flag_empty(self):
        assert self.engine.check_triple_riding([]) is False


# ══════════════════════════════════════════════════════════
# Infraction Detector — Wrong Way / Counterflow Tests
# ══════════════════════════════════════════════════════════

class TestWrongWay:

    def setup_method(self):
        self.engine = TrafficInfractionAnalyzer()

    def _make_box(self, track_id, cx, cy, area=10000):
        return {"label": "motorcycle", "id": track_id, "cx": cx, "cy": cy, "x1": cx - 50, "y1": cy - 50, "x2": cx + 50, "y2": cy + 50}

    def test_no_flag_initially(self):
        assert self.engine.check_wrong_way() is False

    def test_flags_fast_vertical_movement_decreasing(self):
        frame_width = 1280
        cx = 640
        for i in range(15):
            cy = 400 - (i * 12)
            self.engine._update_counterflow_tracks(
                [self._make_box(track_id=1, cx=cx, cy=cy)], frame_width
            )
        assert self.engine.check_wrong_way() is True

    def test_no_flag_for_slow_movement(self):
        frame_width = 1280
        cx = 640
        for i in range(15):
            cy = 100 + (i * 1)
            self.engine._update_counterflow_tracks(
                [self._make_box(track_id=2, cx=cx, cy=cy, area=5000)], frame_width
            )
        assert self.engine.check_wrong_way() is False

    def test_ignores_edge_zone_motorcycles(self):
        frame_width = 1280
        cx = 10
        for i in range(20):
            cy = 400 - (i * 15)
            self.engine._update_counterflow_tracks(
                [self._make_box(track_id=3, cx=cx, cy=cy)],
                frame_width
            )
        assert self.engine.check_wrong_way() is False

    def test_reset_clears_state(self):
        frame_width = 1280
        for i in range(20):
            self.engine._update_counterflow_tracks(
                [self._make_box(1, 640, 400 - i * 15)],
                frame_width
            )
        assert self.engine.check_wrong_way() is True
        self.engine.reset()
        assert self.engine.check_wrong_way() is False


# ══════════════════════════════════════════════════════════
# Citation Calculation Tests
# ══════════════════════════════════════════════════════════

class TestFineCalculation:

    def test_first_offence_no_multiplier(self):
        base, mult, total = compute_infraction_fee(["NO HELMET"], offence_count=1)
        assert base  == 1000
        assert mult  == 1.0
        assert total == 1000

    def test_second_offence_doubles_fine(self):
        _, _, first  = compute_infraction_fee(["NO HELMET"], offence_count=1)
        _, _, second = compute_infraction_fee(["NO HELMET"], offence_count=2)
        assert second == first * 2

    def test_third_offence_triples_fine(self):
        _, _, first = compute_infraction_fee(["NO HELMET"], offence_count=1)
        _, _, third = compute_infraction_fee(["NO HELMET"], offence_count=3)
        assert third == first * 3

    def test_multiple_violations_sum(self):
        base, _, _ = compute_infraction_fee(["NO HELMET", "TRIPLE RIDING"], offence_count=1)
        assert base == 2000

    def test_wrong_way_higher_base(self):
        base, _, _ = compute_infraction_fee(["WRONG WAY"], offence_count=1)
        assert base == 5000

    def test_unknown_violation_fallback_fine(self):
        base, _, _ = compute_infraction_fee(["UNKNOWN_VIOLATION"], offence_count=1)
        assert base == 500


# ══════════════════════════════════════════════════════════
# Plate OCR — Regex Pattern Matching Tests
# ══════════════════════════════════════════════════════════

class TestPlateRegex:

    PLATE_PATTERN = re.compile(r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}')

    def test_valid_standard_plate(self):
        assert self.PLATE_PATTERN.search("KA01AB1234")

    def test_valid_single_letter_series(self):
        assert self.PLATE_PATTERN.search("MH02A5678")

    def test_valid_dl_plate(self):
        assert self.PLATE_PATTERN.search("DL09WR3456")

    def test_rejects_too_short(self):
        assert not self.PLATE_PATTERN.search("KA011")

    def test_rejects_all_letters(self):
        assert not self.PLATE_PATTERN.search("ABCDEFGH")

    def test_rejects_random_string(self):
        assert not self.PLATE_PATTERN.search("INVALID123")

    def test_extracts_from_noisy_string(self):
        noisy = "XYZKA01AB1234ABC"
        m = self.PLATE_PATTERN.search(noisy)
        assert m and m.group() == "KA01AB1234"

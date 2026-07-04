"""
Unit tests for the parts of touchless_controller.py that don't need a real
camera or display -- the geometry/math helpers and the finger-state logic.

The main gesture loop (run()) opens a live camera and a GUI window, so it's
intentionally NOT unit tested here; that's verified manually. CI just makes
sure the underlying math/logic hasn't regressed.
"""

import pytest

from touchless_controller import (
    dist,
    midpoint,
    clamp,
    map_camera_to_screen,
    count_fingers,
    LM,
)


# ── Fakes for MediaPipe hand landmarks ────────────────────────────────────
class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakeHand:
    """A minimal stand-in for MediaPipe's hand landmark list (21 points)."""

    def __init__(self):
        # default everything to the wrist position so unused points don't matter
        self.landmark = [FakeLandmark(0.5, 0.9) for _ in range(21)]

    def set_point(self, idx, x, y):
        self.landmark[idx.value] = FakeLandmark(x, y)
        return self


def make_extended_hand():
    """All 5 fingertips clearly above (smaller y than) their PIP joints."""
    hand = FakeHand()
    pairs = [
        (LM.THUMB_TIP, LM.THUMB_IP),
        (LM.INDEX_FINGER_TIP, LM.INDEX_FINGER_PIP),
        (LM.MIDDLE_FINGER_TIP, LM.MIDDLE_FINGER_PIP),
        (LM.RING_FINGER_TIP, LM.RING_FINGER_PIP),
        (LM.PINKY_TIP, LM.PINKY_PIP),
    ]
    for tip, pip in pairs:
        hand.set_point(pip, 0.5, 0.6)
        hand.set_point(tip, 0.5, 0.2)  # tip well above pip -> "up"
    return hand


def make_curled_hand():
    """All 5 fingertips below their PIP joints (curled into a fist)."""
    hand = FakeHand()
    pairs = [
        (LM.THUMB_TIP, LM.THUMB_IP),
        (LM.INDEX_FINGER_TIP, LM.INDEX_FINGER_PIP),
        (LM.MIDDLE_FINGER_TIP, LM.MIDDLE_FINGER_PIP),
        (LM.RING_FINGER_TIP, LM.RING_FINGER_PIP),
        (LM.PINKY_TIP, LM.PINKY_PIP),
    ]
    for tip, pip in pairs:
        hand.set_point(pip, 0.5, 0.6)
        hand.set_point(tip, 0.5, 0.8)  # tip below pip -> "curled"
    return hand


# ── Geometry helpers ───────────────────────────────────────────────────────


def test_dist_basic():
    assert dist((0, 0), (3, 4)) == 5.0


def test_midpoint_basic():
    assert midpoint((0, 0), (10, 10)) == (5, 5)


def test_clamp_within_range():
    assert clamp(5, 0, 10) == 5


def test_clamp_below_range():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above_range():
    assert clamp(15, 0, 10) == 10


def test_map_camera_to_screen_center():
    # midpoint of the control zone should map to the midpoint of the screen
    x, y = map_camera_to_screen(50, 50, 0, 0, 100, 100, 1920, 1080)
    assert x == pytest.approx(960, abs=1)
    assert y == pytest.approx(540, abs=1)


def test_map_camera_to_screen_clamps_outside_zone():
    # a point far outside the control zone should clamp to the screen edge
    x, y = map_camera_to_screen(-500, -500, 0, 0, 100, 100, 1920, 1080)
    assert x == 0
    assert y == 0


# ── Finger-state detection ─────────────────────────────────────────────────


def test_count_fingers_all_extended():
    hand = make_extended_hand()
    fingers = count_fingers(hand, w=100, h=100)
    assert fingers == [True, True, True, True, True]


def test_count_fingers_all_curled():
    hand = make_curled_hand()
    fingers = count_fingers(hand, w=100, h=100)
    assert fingers == [False, False, False, False, False]

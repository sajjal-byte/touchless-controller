import cv2
import mediapipe as mp
import numpy as np
import time
import math
import sys
import platform
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

# ─── Optional system-control imports (graceful fallback for headless env) ────
try:
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    PYAUTOGUI_OK = True
except Exception:
    PYAUTOGUI_OK = False

try:
    from pynput.keyboard import Key, Controller as KbController

    _keyboard = KbController()
    PYNPUT_OK = True
except Exception:
    PYNPUT_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  ENUMS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════


class Mode(Enum):
    IDLE = auto()
    CURSOR = auto()
    CLICK = auto()
    RIGHT = auto()
    SCROLL = auto()
    SLIDE_NEXT = auto()
    SLIDE_PREV = auto()
    VOLUME = auto()


@dataclass
class AppConfig:
    WINDOW_NAME: str = "Touchless Controller"
    CAM_INDEX: int = 0
    # Lower capture resolution = big speed win. MediaPipe processes every pixel
    # you hand it, and the old 1280x720 default was doing far more work than it
    # needed to for hand tracking. Raise this back up if your machine is fast.
    CAM_W: int = 960
    CAM_H: int = 540
    FLIP: bool = True

    MAX_HANDS: int = 2
    DETECT_CONF: float = 0.75
    TRACK_CONF: float = 0.75
    MODEL_COMPLEXITY: int = 0  # 0 = lite/fast model, 1 = full/slower model

    PINCH_THRESH_PX: float = 45
    CLICK_COOLDOWN: float = 0.45
    SCROLL_SPEED: float = 120.0
    CURSOR_SMOOTH: int = 5
    CURSOR_DEADZONE: float = 4.0
    CONTROL_ZONE_RATIO: float = 0.72
    SLIDE_CENTER_RATIO: float = 0.22
    SLIDE_TRIGGER_RATIO: float = 0.30
    SWIPE_TIME_MAX: float = 0.65
    DISPLAY_SCALE: float = 0.65
    WINDOW_TOPMOST: bool = True
    VOL_SENSITIVITY: float = 0.12

    ACCENT: tuple = (0, 220, 180)
    ACCENT2: tuple = (255, 160, 0)
    DANGER: tuple = (50, 80, 255)
    HUD_W: int = 340
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX


CFG = AppConfig()

# ── MediaPipe aliases ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
LM = mp_hands.HandLandmark
HAND_CONN = mp_hands.HAND_CONNECTIONS


# ══════════════════════════════════════════════════════════════════════════════
#  LANDMARK HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def lm(hand, idx, w, h):
    p = hand.landmark[idx]
    return int(p.x * w), int(p.y * h)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a, b):
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def finger_up(hand, tip_idx, pip_idx, w, h):
    """True if finger tip is above its PIP joint (extended).
    NOTE: this vertical-only test is a reasonable approximation for the four
    fingers, but it is unreliable for the thumb (which moves mostly
    horizontally) and for any finger while it's actively curling into a pinch.
    Gesture logic below avoids relying on this for the specific finger(s)
    that are doing the pinching."""
    _, ty = lm(hand, tip_idx, w, h)
    _, py = lm(hand, pip_idx, w, h)
    return ty < py


def count_fingers(hand, w, h):
    """Returns [thumb, index, middle, ring, pinky] bool list."""
    tip_pip = [
        (LM.THUMB_TIP, LM.THUMB_IP),
        (LM.INDEX_FINGER_TIP, LM.INDEX_FINGER_PIP),
        (LM.MIDDLE_FINGER_TIP, LM.MIDDLE_FINGER_PIP),
        (LM.RING_FINGER_TIP, LM.RING_FINGER_PIP),
        (LM.PINKY_TIP, LM.PINKY_PIP),
    ]
    return [finger_up(hand, t, p, w, h) for t, p in tip_pip]


def hand_label(result, idx):
    try:
        return result.multi_handedness[idx].classification[0].label
    except Exception:
        return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM ACTIONS
# ══════════════════════════════════════════════════════════════════════════════


def sys_move(x, y):
    if PYAUTOGUI_OK:
        try:
            pyautogui.moveTo(x, y, duration=0.0, _pause=False)
        except Exception:
            pyautogui.moveTo(x, y, _pause=False)


def sys_click(button="left"):
    if PYAUTOGUI_OK:
        pyautogui.click(button=button)


def sys_scroll(dy):
    if PYAUTOGUI_OK:
        pyautogui.scroll(int(dy))


def sys_key(key_name):
    if PYNPUT_OK:
        key_map = {
            "right": Key.right,
            "left": Key.left,
            "up": Key.up,
            "down": Key.down,
            "f5": Key.f5,
            "escape": Key.esc,
            "space": Key.space,
        }
        k = key_map.get(key_name.lower())
        if k:
            _keyboard.press(k)
            _keyboard.release(k)
    elif PYAUTOGUI_OK:
        pyautogui.press(key_name)


def sys_volume(delta):
    if PYAUTOGUI_OK:
        presses = max(1, int(abs(delta)))
        if delta > 0:
            pyautogui.press("volumeup", presses=presses)
        else:
            pyautogui.press("volumedown", presses=presses)


def clamp(val, lo, hi):
    return max(lo, min(val, hi))


def map_camera_to_screen(x, y, x1, y1, x2, y2, scr_w, scr_h):
    x = clamp(x, x1, x2)
    y = clamp(y, y1, y2)
    norm_x = (x - x1) / max(1, x2 - x1)
    norm_y = (y - y1) / max(1, y2 - y1)
    return int(norm_x * scr_w), int(norm_y * scr_h)


# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE STATE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GestureDetector:
    mode: Mode = Mode.IDLE
    last_click: float = 0.0
    swipe_origin: Optional[tuple] = None
    swipe_start_t: float = 0.0
    cursor_buf: deque = field(default_factory=lambda: deque(maxlen=CFG.CURSOR_SMOOTH))
    swipe_trail: deque = field(default_factory=lambda: deque(maxlen=28))
    last_vol_y: Optional[int] = None
    status_msg: str = ""
    status_exp: float = 0.0
    pinch_ratio: float = 0.0  # 0=open, 1=fully pinched

    def set_status(self, msg, dur=1.5):
        self.status_msg = msg
        self.status_exp = time.time() + dur

    def get_status(self):
        return self.status_msg if time.time() < self.status_exp else ""

    def smooth_cursor(self, x, y):
        if self.cursor_buf:
            prev_x, prev_y = self.cursor_buf[-1]
            if abs(x - prev_x) <= CFG.CURSOR_DEADZONE and abs(y - prev_y) <= CFG.CURSOR_DEADZONE:
                x, y = prev_x, prev_y
        self.cursor_buf.append((x, y))
        return (
            int(np.mean([p[0] for p in self.cursor_buf])),
            int(np.mean([p[1] for p in self.cursor_buf])),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
#  (blend_rect / draw_hand_skeleton previously copied the ENTIRE frame on every
#  call just to alpha-blend a small rectangle or a hand outline. With the HUD
#  calling blend_rect ~15-20x per frame at 1280x720, that was the single
#  biggest cause of slowdown. Both are rewritten below to only touch the
#  pixels they actually need.)
# ══════════════════════════════════════════════════════════════════════════════


def put_text(img, text, pos, scale=0.52, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, pos, CFG.FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, CFG.FONT, scale, color, thickness, cv2.LINE_AA)


def draw_dot(img, pt, r=7, color=(255, 255, 255)):
    cv2.circle(img, pt, r + 2, (0, 0, 0), -1)
    cv2.circle(img, pt, r, color, -1)


def draw_pinch_arc(img, center, r, progress, color):
    angle = int(360 * min(progress, 1.0))
    if angle > 0:
        cv2.ellipse(img, center, (r, r), -90, 0, angle, color, 3, cv2.LINE_AA)


def blend_rect(img, x1, y1, x2, y2, color, alpha=0.72):
    """Alpha-blend a solid rectangle, operating only on the ROI (not the
    whole frame)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    rect = np.full_like(roi, color)
    cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_hand_skeleton(frame, hand, w, h, color):
    """Draws the skeleton directly onto the frame (no full-frame copy)."""
    pts = {idx: lm(hand, idx, w, h) for idx in range(21)}
    for a, b in HAND_CONN:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    tips = {
        LM.THUMB_TIP.value,
        LM.INDEX_FINGER_TIP.value,
        LM.MIDDLE_FINGER_TIP.value,
        LM.RING_FINGER_TIP.value,
        LM.PINKY_TIP.value,
    }
    for idx, pt in pts.items():
        if idx in tips:
            draw_dot(frame, pt, 9, color)
        elif idx == LM.WRIST.value:
            draw_dot(frame, pt, 10, CFG.ACCENT2)
        else:
            draw_dot(frame, pt, 4, (180, 185, 205))
    tip = pts[LM.INDEX_FINGER_TIP.value]
    cv2.line(frame, (tip[0] - 16, tip[1]), (tip[0] + 16, tip[1]), color, 1, cv2.LINE_AA)
    cv2.line(frame, (tip[0], tip[1] - 16), (tip[0], tip[1] + 16), color, 1, cv2.LINE_AA)


def draw_swipe_trail(frame, trail, color):
    pts = list(trail)
    for i in range(1, len(pts)):
        a = i / len(pts)
        c = tuple(int(v * a) for v in color)
        cv2.line(frame, pts[i - 1], pts[i], c, max(2, int(4 * a)), cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════════════════════

GESTURE_HELP = [
    ("Index only", "Move cursor"),
    ("Index + Thumb pinch", "Left click"),
    ("Ring + Thumb pinch", "Right click"),
    ("Index + Middle", "Scroll"),
    ("3-finger swipe ->", "Next slide"),
    ("3-finger swipe <-", "Prev slide"),
    ("Fist up / down", "Volume up/down"),
]

MODE_COLOR = {
    Mode.IDLE: (100, 100, 110),
    Mode.CURSOR: (0, 220, 180),
    Mode.CLICK: (0, 200, 110),
    Mode.RIGHT: (180, 100, 255),
    Mode.SCROLL: (255, 200, 0),
    Mode.SLIDE_NEXT: (255, 130, 0),
    Mode.SLIDE_PREV: (255, 90, 0),
    Mode.VOLUME: (0, 200, 255),
}

MODE_KEYWORD = {
    Mode.CURSOR: "cursor",
    Mode.CLICK: "click",
    Mode.RIGHT: "right",
    Mode.SCROLL: "scroll",
    Mode.SLIDE_NEXT: "next",
    Mode.SLIDE_PREV: "prev",
    Mode.VOLUME: "volume",
}


def draw_hud(frame, gd: GestureDetector, fps: float, hands: int):
    fh, fw = frame.shape[:2]
    hx = fw - CFG.HUD_W - 10
    hy = 10
    hw = CFG.HUD_W
    hh = fh - 20

    blend_rect(frame, hx, hy, hx + hw, hy + hh, (12, 14, 20), alpha=0.80)
    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (38, 42, 52), 1)

    mc = MODE_COLOR.get(gd.mode, CFG.ACCENT)
    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + 4), mc, -1)

    cy = hy + 28
    put_text(frame, "TOUCHLESS CTRL", (hx + 14, cy), scale=0.68, color=CFG.ACCENT, thickness=1)
    cy += 18
    put_text(
        frame,
        f"v1.1  {platform.system()}",
        (hx + 14, cy),
        scale=0.38,
        color=(70, 72, 90),
        thickness=1,
    )

    cy += 22
    blend_rect(frame, hx + 10, cy - 14, hx + hw - 10, cy + 6, (22, 26, 36), alpha=0.9)
    mode_name = gd.mode.name.replace("_", " ")
    put_text(frame, f"  {mode_name}", (hx + 14, cy), scale=0.52, color=mc, thickness=1)

    cy += 22
    hc = (0, 180, 80) if hands > 0 else (70, 70, 80)
    put_text(frame, f"Hands: {hands}", (hx + 14, cy), scale=0.42, color=hc)
    put_text(frame, f"FPS: {fps:.0f}", (hx + hw - 80, cy), scale=0.42, color=(100, 200, 255))

    cy += 14
    cv2.line(frame, (hx + 10, cy), (hx + hw - 10, cy), (35, 38, 50), 1)
    cy += 14

    put_text(frame, "GESTURES", (hx + 14, cy), scale=0.38, color=(65, 68, 88))
    cy += 16

    kw = MODE_KEYWORD.get(gd.mode, "")
    for icon, desc in GESTURE_HELP:
        active = kw and kw in desc.lower()
        bg = (28, 36, 52) if active else (16, 18, 26)
        tc = (0, 220, 180) if active else (140, 145, 165)
        dc = (200, 202, 215) if active else (90, 93, 110)
        blend_rect(frame, hx + 10, cy - 12, hx + hw - 10, cy + 5, bg, alpha=0.85)
        put_text(frame, icon, (hx + 16, cy), scale=0.40, color=tc)
        put_text(frame, desc, (hx + 150, cy), scale=0.38, color=dc)
        cy += 21

    cy += 4
    cv2.line(frame, (hx + 10, cy), (hx + hw - 10, cy), (35, 38, 50), 1)
    cy += 14

    if gd.mode == Mode.VOLUME:
        put_text(frame, "Fist height = volume", (hx + 14, cy), scale=0.38, color=CFG.ACCENT2)
        cy += 18

    if gd.pinch_ratio > 0.05:
        draw_pinch_arc(frame, (55, fh - 55), 28, gd.pinch_ratio, CFG.ACCENT)
        put_text(frame, "PINCH", (34, fh - 20), scale=0.38, color=CFG.ACCENT)

    status = gd.get_status()
    if status:
        blend_rect(frame, hx + 10, cy - 2, hx + hw - 10, cy + 17, (20, 45, 25), alpha=0.88)
        put_text(frame, status, (hx + 16, cy + 12), scale=0.43, color=(80, 255, 120))
        cy += 24

    put_text(frame, "[Q] Quit  [R] Reset", (hx + 14, hy + hh - 14), scale=0.35, color=(46, 50, 64))

    put_text(frame, "OpenCV + MediaPipe", (12, fh - 12), scale=0.36, color=(42, 46, 58))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════


def run():
    hands_solver = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=CFG.MAX_HANDS,
        min_detection_confidence=CFG.DETECT_CONF,
        min_tracking_confidence=CFG.TRACK_CONF,
        model_complexity=CFG.MODEL_COMPLEXITY,
    )

    cap = cv2.VideoCapture(CFG.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't let old frames queue up
    except Exception:
        pass

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CFG.CAM_INDEX.")
        sys.exit(1)

    cv2.namedWindow(CFG.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        CFG.WINDOW_NAME, int(CFG.CAM_W * CFG.DISPLAY_SCALE), int(CFG.CAM_H * CFG.DISPLAY_SCALE)
    )
    if CFG.WINDOW_TOPMOST:
        cv2.setWindowProperty(CFG.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    try:
        SCR_W, SCR_H = pyautogui.size()
    except Exception:
        SCR_W, SCR_H = 1920, 1080

    gd = GestureDetector()
    fps = 0.0
    fc = 0
    t0 = time.time()

    zone_x1 = zone_y1 = zone_x2 = zone_y2 = None

    print("=" * 58)
    print("  Touchless Controller  |  Press [Q] to quit")
    print("=" * 58)
    if not PYAUTOGUI_OK:
        print("  [!] System control unavailable (no DISPLAY).")
        print("      Gesture DETECTION is fully functional.")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        if CFG.FLIP:
            frame = cv2.flip(frame, 1)

        fh, fw = frame.shape[:2]

        if zone_x1 is None:
            zone_w = int(fw * CFG.CONTROL_ZONE_RATIO)
            zone_h = int(fh * CFG.CONTROL_ZONE_RATIO)
            zone_x1 = (fw - zone_w) // 2
            zone_y1 = (fh - zone_h) // 2
            zone_x2 = zone_x1 + zone_w
            zone_y2 = zone_y1 + zone_h

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands_solver.process(rgb)
        rgb.flags.writeable = True

        fc += 1
        elapsed = time.time() - t0
        if elapsed >= 0.4:
            fps = fc / elapsed
            fc, t0 = 0, time.time()

        gd.mode = Mode.IDLE
        gd.pinch_ratio = 0.0
        n_hands = 0

        if result.multi_hand_landmarks:
            n_hands = len(result.multi_hand_landmarks)

            for hi, hand in enumerate(result.multi_hand_landmarks):
                label = hand_label(result, hi)
                skel_col = CFG.ACCENT if label == "Right" else CFG.ACCENT2
                draw_hand_skeleton(frame, hand, fw, fh, skel_col)

            hand = result.multi_hand_landmarks[0]
            fingers = count_fingers(hand, fw, fh)
            thumb, index, middle, ring, pinky = fingers
            n_up = sum(fingers)

            tip_idx = lm(hand, LM.INDEX_FINGER_TIP, fw, fh)
            tip_thumb = lm(hand, LM.THUMB_TIP, fw, fh)
            tip_mid = lm(hand, LM.MIDDLE_FINGER_TIP, fw, fh)
            tip_ring = lm(hand, LM.RING_FINGER_TIP, fw, fh)
            wrist = lm(hand, LM.WRIST, fw, fh)

            d_it = dist(tip_idx, tip_thumb)  # index-thumb distance
            d_rt = dist(tip_ring, tip_thumb)  # ring-thumb distance

            is_pinch_left = d_it < CFG.PINCH_THRESH_PX
            is_pinch_right = d_rt < CFG.PINCH_THRESH_PX

            # ─────────────────────────────────────────────────────────────
            # Gesture priority chain. IMPORTANT: click gestures are driven by
            # tip-to-tip DISTANCE, not by whether the pinching finger reads
            # as "up" -- a finger curling in to pinch will naturally fail
            # the up/down test, which is exactly why clicks used to never
            # fire. We only use the up/down test on fingers that are NOT
            # part of the pinch, to tell gestures apart.
            # ─────────────────────────────────────────────────────────────

            if is_pinch_right and not index and not middle:
                # Ring + thumb pinch, index & middle folded away -> right click
                gd.mode = Mode.RIGHT
                now = time.time()
                if now - gd.last_click > CFG.CLICK_COOLDOWN:
                    cx, cy = map_camera_to_screen(
                        tip_ring[0], tip_ring[1], zone_x1, zone_y1, zone_x2, zone_y2, SCR_W, SCR_H
                    )
                    sys_move(cx, cy)
                    sys_click("right")
                    gd.last_click = now
                    gd.set_status("Right Click", 0.8)
                mid_pt = midpoint(tip_ring, tip_thumb)
                cv2.circle(frame, mid_pt, 14, (180, 100, 255), 2, cv2.LINE_AA)
                gd.last_vol_y = None
                gd.swipe_trail.clear()

            elif is_pinch_left and not middle and not ring and not pinky:
                # Index + thumb pinch, other fingers folded -> left click
                gd.pinch_ratio = max(0.0, 1.0 - d_it / CFG.PINCH_THRESH_PX)
                gd.mode = Mode.CLICK
                now = time.time()
                if now - gd.last_click > CFG.CLICK_COOLDOWN:
                    cx, cy = map_camera_to_screen(
                        tip_idx[0], tip_idx[1], zone_x1, zone_y1, zone_x2, zone_y2, SCR_W, SCR_H
                    )
                    sys_move(cx, cy)
                    sys_click("left")
                    gd.last_click = now
                    gd.set_status("Left Click", 0.8)
                mid_pt = midpoint(tip_idx, tip_thumb)
                cv2.circle(frame, mid_pt, 14, CFG.ACCENT, 2, cv2.LINE_AA)
                cv2.circle(frame, mid_pt, 4, CFG.ACCENT, -1)
                gd.last_vol_y = None
                gd.swipe_trail.clear()

            elif index and not middle and not ring and not pinky:
                # Plain pointing finger -> cursor move
                gd.mode = Mode.CURSOR
                screen_x, screen_y = map_camera_to_screen(
                    tip_idx[0], tip_idx[1], zone_x1, zone_y1, zone_x2, zone_y2, SCR_W, SCR_H
                )
                sx, sy = gd.smooth_cursor(screen_x, screen_y)
                sys_move(sx, sy)
                gd.swipe_trail.append(tip_idx)
                draw_swipe_trail(frame, gd.swipe_trail, CFG.ACCENT)
                gd.last_vol_y = None

            elif index and middle and not ring and not pinky:
                # Index + middle -> scroll
                gd.mode = Mode.SCROLL
                cv2.line(frame, tip_idx, tip_mid, CFG.ACCENT2, 2, cv2.LINE_AA)
                norm_y = 1.0 - tip_idx[1] / fh
                scroll_v = int((norm_y - 0.5) * CFG.SCROLL_SPEED)
                if abs(scroll_v) >= 1:
                    sys_scroll(scroll_v)
                gd.last_vol_y = None
                gd.swipe_trail.clear()

            elif n_up == 0:
                # Closed fist -> volume, based on wrist height movement
                gd.mode = Mode.VOLUME
                if gd.last_vol_y is not None:
                    dy = gd.last_vol_y - wrist[1]
                    if abs(dy) > 4:
                        delta = dy * CFG.VOL_SENSITIVITY
                        sys_volume(delta)
                        arrow = "up" if dy > 0 else "down"
                        gd.set_status(f"Volume {arrow}", 0.7)
                gd.last_vol_y = wrist[1]
                gd.swipe_trail.clear()

            else:
                gd.last_vol_y = None
                gd.swipe_trail.clear()

            # ── Three-finger center-to-side slide nav (independent check) ──
            if n_up >= 3 and index and middle and ring:
                center_x1 = int(fw * (0.5 - CFG.SLIDE_CENTER_RATIO))
                center_x2 = int(fw * (0.5 + CFG.SLIDE_CENTER_RATIO))
                trigger_left = int(fw * CFG.SLIDE_TRIGGER_RATIO)
                trigger_right = int(fw * (1.0 - CFG.SLIDE_TRIGGER_RATIO))

                if gd.swipe_origin is None:
                    if center_x1 <= tip_idx[0] <= center_x2:
                        gd.swipe_origin = tip_idx
                        gd.swipe_start_t = time.time()
                        gd.set_status("Slide ready", 1.0)
                else:
                    dt = time.time() - gd.swipe_start_t
                    if dt <= CFG.SWIPE_TIME_MAX:
                        if tip_idx[0] >= trigger_right:
                            gd.mode = Mode.SLIDE_NEXT
                            sys_key("right")
                            gd.set_status("Next Slide ->", 1.0)
                            gd.swipe_origin = None
                        elif tip_idx[0] <= trigger_left:
                            gd.mode = Mode.SLIDE_PREV
                            sys_key("left")
                            gd.set_status("<- Prev Slide", 1.0)
                            gd.swipe_origin = None
                    else:
                        gd.swipe_origin = None
            else:
                gd.swipe_origin = None

        else:
            gd.last_vol_y = None
            gd.swipe_trail.clear()
            gd.swipe_origin = None

        draw_hud(frame, gd, fps, n_hands)

        if CFG.DISPLAY_SCALE != 1.0:
            display_frame = cv2.resize(
                frame,
                (int(fw * CFG.DISPLAY_SCALE), int(fh * CFG.DISPLAY_SCALE)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            display_frame = frame
        cv2.imshow(CFG.WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            gd = GestureDetector()
            gd.set_status("Reset", 1.0)

    cap.release()
    cv2.destroyAllWindows()
    hands_solver.close()
    print("\n[Done] Controller stopped.")


if __name__ == "__main__":
    run()

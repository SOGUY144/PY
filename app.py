import time
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import pyautogui  # type: ignore
except Exception:
    pyautogui = None

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def landmarks_to_array(hand_landmarks, w, h):
    pts = []
    for lm in hand_landmarks:
        pts.append([lm.x * w, lm.y * h])
    return np.array(pts, dtype=np.float32)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def motion_to_color_bgr(motion, slow=20.0, fast=450.0):
    t = clamp((motion - slow) / (fast - slow), 0.0, 1.0)
    b = int(lerp(255, 0, t))
    g = int(lerp(80, 40, t))
    r = int(lerp(0, 255, t))
    return (b, g, r)


def alpha_blend(dst, overlay, alpha):
    cv2.addWeighted(overlay, alpha, dst, 1 - alpha, 0, dst)


def draw_glow_line(overlay, p1, p2, color, thickness):
    for t, a in ((thickness + 10, 0.08), (thickness + 6, 0.12), (thickness + 2, 0.18)):
        cv2.line(overlay, p1, p2, color, t, cv2.LINE_AA)
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)


def draw_glow_circle(overlay, center, radius, color):
    for r, a in ((radius + 10, 0.06), (radius + 6, 0.10), (radius + 3, 0.16)):
        cv2.circle(overlay, center, r, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, center, radius, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)


def draw_hand_ironman(frame, pts, color, alpha=0.45):
    overlay = frame.copy()
    for a, b in HAND_CONNECTIONS:
        p1 = (int(pts[a, 0]), int(pts[a, 1]))
        p2 = (int(pts[b, 0]), int(pts[b, 1]))
        draw_glow_line(overlay, p1, p2, color, 2)
    for x, y in pts:
        draw_glow_circle(overlay, (int(x), int(y)), 3, color)
    alpha_blend(frame, overlay, alpha)


def draw_hand_clean(frame, pts, color):
    for a, b in HAND_CONNECTIONS:
        p1 = (int(pts[a, 0]), int(pts[a, 1]))
        p2 = (int(pts[b, 0]), int(pts[b, 1]))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def finger_states(norm_lms):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    states = []
    for tip, pip in zip(tips, pips):
        states.append(norm_lms[tip].y < norm_lms[pip].y)
    return states


def classify_gesture(norm_lms):
    thumb, index_, middle, ring, pinky = finger_states(norm_lms)
    if index_ and middle and (not ring) and (not pinky):
        return "Peace"
    if (not index_) and (not middle) and (not ring) and (not pinky):
        return "Fist"
    return ""


def palm_center_px(pts):
    center = np.mean(pts[[0, 5, 9, 13, 17]], axis=0)
    return float(center[0]), float(center[1])


def estimate_depth_scale(pts):
    palm = np.mean(pts[[0, 5, 9, 13, 17]], axis=0)
    d = float(np.linalg.norm(pts[0] - palm))
    scale = clamp((d - 35.0) / (180.0 - 35.0), 0.0, 1.0)
    return scale


def draw_hud(frame, pts, color, depth_scale, angle_deg):
    overlay = frame.copy()
    cx, cy = palm_center_px(pts)
    cx_i, cy_i = int(cx), int(cy)
    base_r = int(lerp(32, 78, depth_scale))
    ring_color = (color[0], min(255, color[1] + 40), min(255, color[2] + 40))
    cv2.ellipse(overlay, (cx_i, cy_i), (base_r, base_r), angle_deg, 0, 360, ring_color, 2, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx_i, cy_i), (base_r + 10, base_r + 10), -angle_deg * 0.6, 20, 160, ring_color, 2, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx_i, cy_i), (base_r + 10, base_r + 10), -angle_deg * 0.6, 200, 340, ring_color, 2, cv2.LINE_AA)

    tick_len = int(lerp(10, 22, depth_scale))
    rad = math.radians(angle_deg)
    x2 = int(cx + math.cos(rad) * (base_r + tick_len))
    y2 = int(cy + math.sin(rad) * (base_r + tick_len))
    draw_glow_line(overlay, (cx_i, cy_i), (x2, y2), ring_color, 2)
    alpha_blend(frame, overlay, 0.28)


def draw_bbox(frame, pts, color):
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    x1, y1 = int(min_xy[0]), int(min_xy[1])
    x2, y2 = int(max_xy[0]), int(max_xy[1])
    pad = 14
    x1, y1 = x1 - pad, y1 - pad
    x2, y2 = x2 + pad, y2 + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    alpha_blend(frame, overlay, 0.25)
    return (x1, y1, x2, y2)


def apply_background_blur(frame_bgr, mask_float, blur_ksize=31):
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    m = np.clip(mask_float, 0.0, 1.0).astype(np.float32)
    if m.ndim == 2:
        m = m[:, :, None]
    out = frame_bgr.astype(np.float32) * m + blurred.astype(np.float32) * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_face_hud(frame, face_lms, color):
    h, w = frame.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    x1, y1 = int(min_xy[0]) - 12, int(min_xy[1]) - 12
    x2, y2 = int(max_xy[0]) + 12, int(max_xy[1]) + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    alpha_blend(frame, overlay, 0.20)

    for idx in (1, 33, 263, 61, 291, 199):
        if idx < len(pts):
            draw_glow_circle(frame, (int(pts[idx, 0]), int(pts[idx, 1])), 3, color)
    return (x1, y1, x2, y2)


def draw_text_panel(frame, origin, lines, color, scale=0.55):
    x, y = origin
    overlay = frame.copy()
    width = 0
    heights = []
    for s in lines:
        (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        width = max(width, tw)
        heights.append(th)
    panel_w = width + 18
    panel_h = int(sum(heights) + 14 + 8 * (len(lines) - 1))
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (8, 8, 18), -1)
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), color, 1)
    alpha_blend(frame, overlay, 0.35)
    ty = y + 18
    for s in lines:
        cv2.putText(frame, s, (x + 9, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        ty += int(18 + 8 * (scale - 0.45))


def states_only_index(norm_lms):
    thumb, index_, middle, ring, pinky = finger_states(norm_lms)
    return index_ and (not middle) and (not ring) and (not pinky)


def smooth_point(prev, curr, alpha=0.22):
    if prev is None:
        return curr
    return (prev[0] + (curr[0] - prev[0]) * alpha, prev[1] + (curr[1] - prev[1]) * alpha)


class ARWindow:
    def __init__(self, x, y, w, h, title="AR", alpha=0.25):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.title = title
        self.alpha = float(alpha)

        self.is_dragging = False
        self._drag_offset = (0.0, 0.0)
        self._target = (self.x, self.y)

    def contains(self, px, py):
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

    def start_drag(self, finger_xy):
        fx, fy = finger_xy
        self.is_dragging = True
        self._drag_offset = (fx - self.x, fy - self.y)

    def stop_drag(self):
        self.is_dragging = False

    def update(self, finger_xy, pinch, smoothing=0.25):
        fx, fy = finger_xy

        if self.is_dragging and pinch:
            tx = fx - self._drag_offset[0]
            ty = fy - self._drag_offset[1]
            self._target = (tx, ty)
        elif self.is_dragging and (not pinch):
            self.stop_drag()

        self.x = lerp(self.x, self._target[0], smoothing)
        self.y = lerp(self.y, self._target[1], smoothing)

    def draw(self, frame, color, highlight=False):
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = int(self.x + self.w), int(self.y + self.h)

        overlay = frame.copy()
        fill = (8, 8, 18)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        alpha_blend(frame, overlay, self.alpha)

        border = color
        if highlight:
            border = (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60))

        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2, cv2.LINE_AA)
        cv2.putText(frame, self.title, (x1 + 10, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border, 2, cv2.LINE_AA)
        return (x1, y1, x2, y2)


def draw_mode_bar(frame, mode, mouse_enabled):
    names = {
        1: "HUD",
        2: "MOUSE",
        3: "CANVAS",
        4: "3D",
        5: "FACE",
    }
    text = f"[{mode}] {names.get(mode, '?')}"
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 245), 2, cv2.LINE_AA)
    if mode == 2:
        st = "ON" if mouse_enabled else "OFF"
        cv2.putText(frame, f"Mouse {st} (M)", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 245), 2, cv2.LINE_AA)


def draw_canvas_ui(frame, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 58), (310, 118), (8, 8, 18), -1)
    cv2.rectangle(overlay, (10, 58), (310, 118), color, 1)
    alpha_blend(frame, overlay, 0.25)
    cv2.putText(frame, "Canvas: draw with index-only", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "C change color | X clear", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def project_points(points3d, w, h, fov=520.0, zoff=2.8):
    out = []
    for x, y, z in points3d:
        zz = z + zoff
        sx = (x * fov) / zz + w * 0.5
        sy = (y * fov) / zz + h * 0.5
        out.append((int(sx), int(sy)))
    return out


def rotate_xyz(p, rx, ry, rz):
    x, y, z = p
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    y, z = y * cx - z * sx, y * sx + z * cx
    x, z = x * cy + z * sy, -x * sy + z * cy
    x, y = x * cz - y * sz, x * sz + y * cz
    return (x, y, z)


def draw_cube(frame, rot, scale, color):
    h, w = frame.shape[:2]
    s = float(scale)
    base = [
        (-s, -s, -s),
        (s, -s, -s),
        (s, s, -s),
        (-s, s, -s),
        (-s, -s, s),
        (s, -s, s),
        (s, s, s),
        (-s, s, s),
    ]
    pts = [rotate_xyz(p, rot[0], rot[1], rot[2]) for p in base]
    p2 = project_points(pts, w, h)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    overlay = frame.copy()
    for a, b in edges:
        draw_glow_line(overlay, p2[a], p2[b], color, 2)
    alpha_blend(frame, overlay, 0.25)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera (VideoCapture(0)). Try changing index to 1/2.")

    model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
    if not model_path.exists():
        raise RuntimeError(
            "Missing model file: hand_landmarker.task. Download the MediaPipe Hand Landmarker model "
            "and place it next to app.py."
        )

    prev_pts = None
    prev_time = time.time()
    trail = deque(maxlen=36)

    mode = 1

    mouse_enabled = False
    mouse_prev = None
    click_latch = False
    drag_down = False
    scroll_prev_y = None

    canvas = None
    last_draw_pt = None
    brush_colors = [
        (255, 100, 0),
        (255, 60, 240),
        (80, 220, 255),
        (120, 255, 120),
        (30, 30, 255),
    ]
    brush_i = 0

    cube_rot = [0.0, 0.0, 0.0]

    ar_window = ARWindow(80, 120, 340, 180, title="AR PANEL", alpha=0.22)
    pinch_state = False

    theme_colors = [
        (255, 90, 0),
        (255, 60, 240),
        (80, 220, 255),
        (120, 255, 120),
        (0, 180, 255),
    ]
    theme_i = 2
    use_dynamic_color = False
    use_glow = False
    show_panels = False
    show_bbox = False
    show_trail = False
    show_ring = False

    seg_model_path = Path(__file__).resolve().parent / "selfie_segmenter.tflite"
    face_model_path = Path(__file__).resolve().parent / "face_landmarker.task"
    segmenter = None
    face_landmarker = None

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            now = time.time()
            dt = max(now - prev_time, 1e-6)
            prev_time = now

            motion = 0.0
            pinch_px = 0.0
            pinch_ratio = 0.0
            pinch = False
            depth_scale = 0.0
            gesture = ""
            avg_xyz = (0.0, 0.0, 0.0)
            angle_deg = 0.0
            color = theme_colors[theme_i]
            pts = None
            hand_lms = None
            palm_xy = None
            index_tip = None
            thumb_tip = None
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_lms = result.hand_landmarks[0]
                pts = landmarks_to_array(hand_lms, w, h)
                gesture = classify_gesture(hand_lms)
                cx, cy = palm_center_px(pts)
                palm_xy = (cx, cy)

                if prev_pts is not None and prev_pts.shape == pts.shape:
                    d = np.linalg.norm(pts - prev_pts, axis=1)
                    motion = float(np.mean(d) / dt)
                prev_pts = pts

                if use_dynamic_color:
                    color = motion_to_color_bgr(motion)
                if gesture == "Fist":
                    color = (0, 40, 255)

                depth_scale = estimate_depth_scale(pts)

                v = pts[17] - pts[5]
                angle_deg = math.degrees(math.atan2(float(v[1]), float(v[0])))

                thumb_tip = pts[4]
                index_tip = pts[8]
                pinch_px = float(np.linalg.norm(index_tip - thumb_tip))
                palm_size = float(np.linalg.norm(pts[0] - np.mean(pts[[5, 9, 13, 17]], axis=0)))
                palm_size = max(palm_size, 1.0)
                pinch_ratio = clamp((pinch_px / palm_size - 0.35) / (1.35 - 0.35), 0.0, 1.0)

                pinch_on = pinch_ratio < 0.18
                pinch_off = pinch_ratio > 0.28
                if (not pinch_state) and pinch_on:
                    pinch_state = True
                elif pinch_state and pinch_off:
                    pinch_state = False
                pinch = pinch_state

                avg_x = float(np.mean([lm.x for lm in hand_lms]))
                avg_y = float(np.mean([lm.y for lm in hand_lms]))
                avg_z = float(np.mean([lm.z for lm in hand_lms]))
                avg_xyz = (avg_x, avg_y, avg_z)

            else:
                prev_pts = None

            if mode == 1:
                if pts is not None:
                    if use_glow:
                        draw_hand_ironman(frame, pts, color)
                    else:
                        draw_hand_clean(frame, pts, color)
                    if show_ring:
                        draw_hud(frame, pts, color, depth_scale, angle_deg)
                    if show_bbox:
                        x1, y1, x2, y2 = draw_bbox(frame, pts, color)
                    if show_trail:
                        idx = (int(pts[8, 0]), int(pts[8, 1]))
                        trail.append((idx, time.time(), color))
                    if show_panels and show_bbox:
                        draw_text_panel(
                            frame,
                            (max(8, x1), max(8, y1 - 70)),
                            [
                                f"Gesture: {gesture or '-'}",
                                f"Pinch: {pinch_ratio * 100:.0f}%",
                                f"XYZ: {avg_xyz[0]:.2f}, {avg_xyz[1]:.2f}, {avg_xyz[2]:.3f}",
                            ],
                            color,
                            scale=float(lerp(0.45, 0.75, depth_scale)),
                        )

                if pts is not None and index_tip is not None:
                    fx, fy = float(index_tip[0]), float(index_tip[1])
                    hovered = ar_window.contains(fx, fy)

                    if hovered and pinch and (not ar_window.is_dragging):
                        ar_window.start_drag((fx, fy))

                    ar_window.update((fx, fy), pinch=pinch, smoothing=0.28)
                    ar_window.draw(frame, color, highlight=hovered or ar_window.is_dragging)

                    if ar_window.is_dragging and thumb_tip is not None:
                        tpx, tpy = int(thumb_tip[0]), int(thumb_tip[1])
                        ipx, ipy = int(index_tip[0]), int(index_tip[1])
                        overlay = frame.copy()
                        draw_glow_line(overlay, (tpx, tpy), (ipx, ipy), color, 2)
                        alpha_blend(frame, overlay, 0.35)
                        cv2.putText(
                            frame,
                            f"{pinch_px:.0f}px",
                            (min(tpx, ipx) + 10, min(tpy, ipy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

            elif mode == 2:
                if pts is not None:
                    if use_glow:
                        draw_hand_ironman(frame, pts, color)
                    else:
                        draw_hand_clean(frame, pts, color)
                    if show_bbox:
                        x1, y1, x2, y2 = draw_bbox(frame, pts, color)
                    if show_panels and show_bbox:
                        draw_text_panel(
                            frame,
                            (max(8, x1), max(8, y1 - 70)),
                            [
                                f"Gesture: {gesture or '-'}",
                                f"Pinch: {pinch_ratio * 100:.0f}%",
                                f"Motion: {motion:.0f}px/s",
                            ],
                            color,
                            scale=0.55,
                        )

                    if pyautogui is None:
                        cv2.putText(frame, "pyautogui missing. pip install -r requirements.txt", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        if mouse_enabled:
                            sw, sh = pyautogui.size()
                            tx = int(clamp(pts[8, 0] / w, 0.0, 1.0) * (sw - 1))
                            ty = int(clamp(pts[8, 1] / h, 0.0, 1.0) * (sh - 1))
                            mouse_prev = smooth_point(mouse_prev, (tx, ty), alpha=0.25)
                            pyautogui.moveTo(int(mouse_prev[0]), int(mouse_prev[1]), _pause=False)

                            pinch_on = pinch_ratio < 0.18
                            if pinch_on and (not click_latch):
                                pyautogui.click()
                                click_latch = True
                            if (not pinch_on) and click_latch:
                                click_latch = False

                            if gesture == "Fist" and (not drag_down):
                                pyautogui.mouseDown()
                                drag_down = True
                            if gesture != "Fist" and drag_down:
                                pyautogui.mouseUp()
                                drag_down = False

                            if gesture == "Peace" and palm_xy is not None:
                                if scroll_prev_y is None:
                                    scroll_prev_y = palm_xy[1]
                                dy = palm_xy[1] - scroll_prev_y
                                scroll_prev_y = palm_xy[1]
                                sc = int(clamp(-dy / 6.0, -40, 40))
                                if sc != 0:
                                    pyautogui.scroll(sc)
                            else:
                                scroll_prev_y = None

            elif mode == 3:
                if canvas is None:
                    canvas = np.zeros_like(frame)
                if pts is not None and hand_lms is not None:
                    if use_glow:
                        draw_hand_ironman(frame, pts, color)
                    else:
                        draw_hand_clean(frame, pts, color)
                    if show_panels:
                        draw_canvas_ui(frame, brush_colors[brush_i])
                    if states_only_index(hand_lms):
                        p = (int(pts[8, 0]), int(pts[8, 1]))
                        if last_draw_pt is None:
                            last_draw_pt = p
                        cv2.line(canvas, last_draw_pt, p, brush_colors[brush_i], 6, cv2.LINE_AA)
                        last_draw_pt = p
                    else:
                        last_draw_pt = None
                if canvas is not None:
                    alpha_blend(frame, cv2.addWeighted(frame, 0.35, canvas, 1.0, 0), 0.75)

            elif mode == 4:
                if pts is not None:
                    if use_glow:
                        draw_hand_ironman(frame, pts, color)
                    else:
                        draw_hand_clean(frame, pts, color)
                    cube_rot[0] = lerp(cube_rot[0], (angle_deg / 180.0) * 1.6, 0.18)
                    cube_rot[1] = lerp(cube_rot[1], (pinch_ratio - 0.5) * 2.0, 0.18)
                    cube_rot[2] = lerp(cube_rot[2], (motion / 450.0) * 1.2, 0.10)
                    cube_scale = lerp(0.55, 1.35, depth_scale)
                    draw_cube(frame, cube_rot, cube_scale, color)
                    cv2.putText(frame, "3D: pinch changes yaw | distance changes size", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 240), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Show hand to control cube", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 240), 2, cv2.LINE_AA)

            elif mode == 5:
                if segmenter is None and seg_model_path.exists():
                    base = mp_python.BaseOptions(model_asset_path=str(seg_model_path))
                    seg_opts = vision.ImageSegmenterOptions(
                        base_options=base,
                        running_mode=vision.RunningMode.IMAGE,
                        output_category_mask=True,
                    )
                    segmenter = vision.ImageSegmenter.create_from_options(seg_opts)

                if face_landmarker is None and face_model_path.exists():
                    base = mp_python.BaseOptions(model_asset_path=str(face_model_path))
                    face_opts = vision.FaceLandmarkerOptions(
                        base_options=base,
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=1,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                    face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

                if segmenter is None:
                    cv2.putText(frame, "Segmentation model missing: selfie_segmenter.tflite", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                    seg_res = segmenter.segment(img)
                    mask = seg_res.category_mask.numpy_view()
                    frame = apply_background_blur(frame, mask, blur_ksize=31)

                if face_landmarker is None:
                    cv2.putText(frame, "Face model optional: face_landmarker.task", (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 240), 2, cv2.LINE_AA)
                else:
                    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                    face_res = face_landmarker.detect(img)
                    if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
                        face_color = (80, 220, 255)
                        draw_face_hud(frame, face_res.face_landmarks[0], face_color)

            if len(trail) > 0:
                overlay = frame.copy()
                now_t = time.time()
                for i, (p, t0, c0) in enumerate(trail):
                    age = now_t - t0
                    a = clamp(1.0 - age / 0.45, 0.0, 1.0)
                    if a <= 0:
                        continue
                    rad = int(lerp(10, 3, i / max(1, len(trail) - 1)))
                    col = (min(255, c0[0] + 30), min(255, c0[1] + 30), min(255, c0[2] + 30))
                    cv2.circle(overlay, p, rad, col, -1, cv2.LINE_AA)
                alpha_blend(frame, overlay, 0.20)

            if show_panels:
                cv2.putText(
                    frame,
                    f"motion {motion:.0f}",
                    (12, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            draw_mode_bar(frame, mode, mouse_enabled)

            cv2.imshow("Hand Motion", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                mode = int(chr(key))
            if key in (ord("m"), ord("M")):
                mouse_enabled = not mouse_enabled
                if not mouse_enabled and pyautogui is not None:
                    if drag_down:
                        pyautogui.mouseUp()
                        drag_down = False
            if key in (ord("c"), ord("C")):
                brush_i = (brush_i + 1) % len(brush_colors)
            if key in (ord("x"), ord("X")):
                if canvas is not None:
                    canvas[:] = 0
            if key in (ord("t"), ord("T")):
                theme_i = (theme_i + 1) % len(theme_colors)
            if key in (ord("d"), ord("D")):
                use_dynamic_color = not use_dynamic_color
            if key in (ord("g"), ord("G")):
                use_glow = not use_glow
            if key in (ord("h"), ord("H")):
                show_panels = not show_panels
            if key in (ord("b"), ord("B")):
                show_bbox = not show_bbox
            if key in (ord("p"), ord("P")):
                show_trail = not show_trail
                if not show_trail:
                    trail.clear()
            if key in (ord("r"), ord("R")):
                show_ring = not show_ring
    finally:
        landmarker.close()
        if segmenter is not None:
            segmenter.close()
        if face_landmarker is not None:
            face_landmarker.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

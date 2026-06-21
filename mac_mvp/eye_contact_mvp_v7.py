"""macOS Eye Contact MVP V7 - stabilized full-eye replacement.

V7 keeps the V6 full-eye patch idea but focuses on reducing shakiness:

- smooths iris points
- smooths correction vectors, not just landmarks
- uses conservative default shift
- uses non-destructive patch mode by default
- applies a small motion dead-zone so tiny frame-to-frame changes do not move the eye

Run:
    python mac_mvp/eye_contact_mvp_v7.py --debug

Recommended test without overlays:
    python mac_mvp/eye_contact_mvp_v7.py --strength 0.45 --max-shift-ratio 0.10

Keys:
    q or Esc    quit
    d           toggle debug overlays
    c           toggle correction on/off
    w/s/a/l     move manual target up/down/left/right
    r           reset manual target
    + / -       increase/decrease correction strength
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None

from eye_contact_mvp_v2 import (
    DEFAULT_MODEL_PATH,
    LEFT_EYE_OUTLINE,
    LEFT_IRIS,
    RIGHT_EYE_OUTLINE,
    RIGHT_IRIS,
    LandmarkProvider,
    Stabilizer,
    _bounding_box,
    _draw_debug,
    _iris_center,
    _points_from_landmarks,
)

from eye_contact_mvp_v6 import _copy_eye_patch, _paste_patch, _fill_eye_region


class VectorSmoother:
    def __init__(self, alpha: float, deadzone: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 0.98))
        self.deadzone = max(0.0, deadzone)
        self.state: Dict[str, np.ndarray] = {}

    def update(self, key: str, vector: Tuple[float, float]) -> Tuple[float, float]:
        value = np.asarray(vector, dtype=np.float32)
        previous = self.state.get(key)
        if previous is None:
            self.state[key] = value
            return float(value[0]), float(value[1])

        if np.linalg.norm(value - previous) < self.deadzone:
            return float(previous[0]), float(previous[1])

        smoothed = self.alpha * previous + (1.0 - self.alpha) * value
        self.state[key] = smoothed
        return float(smoothed[0]), float(smoothed[1])


def _eye_center(eye_points: np.ndarray) -> Tuple[float, float]:
    return float(np.median(eye_points[:, 0])), float(np.median(eye_points[:, 1]))


def _draw_hud(frame, lines, face_detected: bool) -> None:
    bg_color = (30, 120, 30) if face_detected else (30, 30, 180)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 104), bg_color, -1)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def _stable_whole_eye_replace(
    frame: np.ndarray,
    eye_key: str,
    box,
    eye_points: np.ndarray,
    iris_center: Optional[Tuple[float, float]],
    vector_smoother: VectorSmoother,
    strength: float,
    target_offset_x: float,
    target_offset_y: float,
    max_shift_ratio: float,
    replace_mode: str,
    mask_scale: float,
    mask_blur: int,
    motion_deadzone: float,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if iris_center is None or not box.valid():
        return None

    copied = _copy_eye_patch(frame, box, eye_points, mask_scale, mask_blur)
    if copied is None:
        return None
    patch, mask = copied

    _, _, eye_w, eye_h = cv2.boundingRect(eye_points)
    if eye_w < 10 or eye_h < 6:
        return None

    eye_cx, eye_cy = _eye_center(eye_points)
    target_x = eye_cx + target_offset_x * eye_w
    target_y = eye_cy + target_offset_y * eye_h

    raw_dx = (target_x - iris_center[0]) * strength
    raw_dy = (target_y - iris_center[1]) * strength

    raw_dx = float(np.clip(raw_dx, -eye_w * max_shift_ratio, eye_w * max_shift_ratio))
    raw_dy = float(np.clip(raw_dy, -eye_h * max_shift_ratio, eye_h * max_shift_ratio))

    if abs(raw_dx) < motion_deadzone:
        raw_dx = 0.0
    if abs(raw_dy) < motion_deadzone:
        raw_dy = 0.0

    dx, dy = vector_smoother.update(eye_key, (raw_dx, raw_dy))

    # Avoid modifying the frame when the correction is effectively zero.
    if abs(dx) < 0.35 and abs(dy) < 0.35:
        start = (int(round(iris_center[0])), int(round(iris_center[1])))
        return start, start

    _fill_eye_region(frame, box, eye_points, mask_scale, replace_mode)
    _paste_patch(frame, patch, mask, box, dx, dy)

    start = (int(round(iris_center[0])), int(round(iris_center[1])))
    end = (int(round(iris_center[0] + dx)), int(round(iris_center[1] + dy)))
    return start, end


def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS camera permissions and camera index.")

    provider = LandmarkProvider(args.landmarker_model)
    point_stabilizer = Stabilizer(alpha=args.point_smoothing, deadzone=args.point_deadzone)
    vector_smoother = VectorSmoother(alpha=args.vector_smoothing, deadzone=args.vector_deadzone)
    print(f"Using MediaPipe backend: {provider.backend}")

    debug = args.debug
    correction_enabled = not args.no_correction
    virtual_enabled = args.virtual_camera
    strength = args.strength
    target_offset_x = args.target_offset_x
    target_offset_y = args.target_offset_y
    current_fps = 0.0
    fps_count = 0
    fps_time = time.time()
    virtual_cam = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = provider.process(rgb)
            face_detected = landmarks is not None
            iris_status = "iris=missing"
            correction_vectors = []

            if landmarks:
                left_eye_points = _points_from_landmarks(landmarks, LEFT_EYE_OUTLINE, width, height)
                right_eye_points = _points_from_landmarks(landmarks, RIGHT_EYE_OUTLINE, width, height)
                left_box = _bounding_box(left_eye_points, width, height, pad_ratio=args.box_padding)
                right_box = _bounding_box(right_eye_points, width, height, pad_ratio=args.box_padding)

                left_iris = _iris_center(landmarks, LEFT_IRIS, width, height)
                right_iris = _iris_center(landmarks, RIGHT_IRIS, width, height)
                iris_status = f"iris=left:{left_iris is not None} right:{right_iris is not None}"

                if left_iris is not None:
                    left_iris = point_stabilizer.update("left_iris", left_iris)
                if right_iris is not None:
                    right_iris = point_stabilizer.update("right_iris", right_iris)

                if correction_enabled:
                    left_vec = _stable_whole_eye_replace(
                        frame, "left_vec", left_box, left_eye_points, left_iris, vector_smoother,
                        strength, target_offset_x, target_offset_y, args.max_shift_ratio,
                        args.replace_mode, args.mask_scale, args.mask_blur, args.motion_deadzone,
                    )
                    right_vec = _stable_whole_eye_replace(
                        frame, "right_vec", right_box, right_eye_points, right_iris, vector_smoother,
                        strength, target_offset_x, target_offset_y, args.max_shift_ratio,
                        args.replace_mode, args.mask_scale, args.mask_blur, args.motion_deadzone,
                    )
                    correction_vectors = [v for v in (left_vec, right_vec) if v is not None]

                if debug:
                    _draw_debug(frame, landmarks, width, height)
                    cv2.rectangle(frame, (left_box.x1, left_box.y1), (left_box.x2, left_box.y2), (255, 255, 255), 1)
                    cv2.rectangle(frame, (right_box.x1, right_box.y1), (right_box.x2, right_box.y2), (255, 255, 255), 1)
                    for start, end in correction_vectors:
                        cv2.arrowedLine(frame, start, end, (0, 0, 255), 2, tipLength=0.35)

            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_time = time.time()

            _draw_hud(
                frame,
                [
                    f"v7 face={'detected' if face_detected else 'missing'} {iris_status} fps={current_fps:.1f}",
                    f"stable-eye replace={args.replace_mode} correction={'on' if correction_enabled else 'off'} strength={strength:.2f}",
                    f"target=({target_offset_x:+.2f},{target_offset_y:+.2f}) vector_smooth={args.vector_smoothing:.2f} keys: w/s/a/l c d r +/-",
                ],
                face_detected,
            )

            if virtual_enabled:
                if pyvirtualcam is None:
                    virtual_enabled = False
                    print("pyvirtualcam is not installed; continuing preview only.")
                elif virtual_cam is None:
                    virtual_cam = pyvirtualcam.Camera(width=width, height=height, fps=args.fps)
                    print(f"Virtual camera started: {virtual_cam.device}")
                else:
                    virtual_cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    virtual_cam.sleep_until_next_frame()

            cv2.imshow("Mac Eye Contact MVP V7", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("c"):
                correction_enabled = not correction_enabled
            if key == ord("v"):
                virtual_enabled = not virtual_enabled
                if not virtual_enabled and virtual_cam is not None:
                    virtual_cam.close()
                    virtual_cam = None
            if key == ord("r"):
                target_offset_x = 0.0
                target_offset_y = 0.0
            if key == ord("w"):
                target_offset_y -= 0.02
            if key == ord("s"):
                target_offset_y += 0.02
            if key == ord("a"):
                target_offset_x -= 0.02
            if key == ord("l"):
                target_offset_x += 0.02
            if key in (ord("+"), ord("=")):
                strength = min(1.4, strength + 0.05)
            if key in (ord("-"), ord("_")):
                strength = max(0.0, strength - 0.05)
    finally:
        if virtual_cam is not None:
            virtual_cam.close()
        provider.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilized full-eye replacement macOS eye-contact MVP V7")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.45)
    parser.add_argument("--point-smoothing", type=float, default=0.88)
    parser.add_argument("--point-deadzone", type=float, default=1.2)
    parser.add_argument("--vector-smoothing", type=float, default=0.92)
    parser.add_argument("--vector-deadzone", type=float, default=0.8)
    parser.add_argument("--motion-deadzone", type=float, default=1.0)
    parser.add_argument("--max-shift-ratio", type=float, default=0.10)
    parser.add_argument("--target-offset-x", type=float, default=0.0)
    parser.add_argument("--target-offset-y", type=float, default=0.0)
    parser.add_argument("--replace-mode", choices=["none", "patch", "inpaint"], default="patch")
    parser.add_argument("--box-padding", type=float, default=0.42)
    parser.add_argument("--mask-scale", type=float, default=1.25)
    parser.add_argument("--mask-blur", type=int, default=25)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--virtual-camera", action="store_true")
    parser.add_argument("--landmarker-model", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

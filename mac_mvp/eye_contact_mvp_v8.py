"""macOS Eye Contact MVP V8 - calibrated gaze correction experiment.

V8 replaces the manual target-only behavior with a simple calibration flow:

1. Look directly into the camera.
2. Press `k` to capture your camera-looking iris position.
3. Look at the screen or elsewhere.
4. The script estimates how far your iris moved away from the camera-looking
   baseline and tries to move the eye patch back toward that baseline.

This is still classical OpenCV patch rendering, not neural eye synthesis, but it
is the first version that tries to adjust automatically based on a calibrated
camera-gaze baseline.

Run:
    python mac_mvp/eye_contact_mvp_v8.py --debug

Keys:
    k           calibrate while looking directly at camera
    c           toggle correction on/off
    d           toggle debug overlays
    q or Esc    quit
    + / -       increase/decrease correction strength
    [ / ]       decrease/increase max shift
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


@dataclass
class EyeState:
    box: object
    eye_points: np.ndarray
    iris: Optional[Tuple[float, float]]
    ratio: Optional[Tuple[float, float]]


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


def _draw_hud(frame, lines, calibrated: bool, face_detected: bool) -> None:
    if not face_detected:
        bg = (30, 30, 180)
    elif calibrated:
        bg = (30, 120, 30)
    else:
        bg = (0, 125, 180)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), bg, -1)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        y += 27


def _iris_ratio(eye_points: np.ndarray, iris: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if iris is None:
        return None
    x, y, w, h = cv2.boundingRect(eye_points)
    if w <= 1 or h <= 1:
        return None
    rx = (iris[0] - x) / float(w)
    ry = (iris[1] - y) / float(h)
    return float(rx), float(ry)


def _extract_eye_state(landmarks, eye_indices, iris_indices, width: int, height: int, stabilizer: Stabilizer, key: str) -> EyeState:
    eye_points = _points_from_landmarks(landmarks, eye_indices, width, height)
    box = _bounding_box(eye_points, width, height, pad_ratio=0.42)
    iris = _iris_center(landmarks, iris_indices, width, height)
    if iris is not None:
        iris = stabilizer.update(f"{key}_iris", iris)
    ratio = _iris_ratio(eye_points, iris)
    return EyeState(box=box, eye_points=eye_points, iris=iris, ratio=ratio)


def _replace_eye_to_calibrated_ratio(
    frame: np.ndarray,
    eye_key: str,
    eye: EyeState,
    calibrated_ratio: Optional[Tuple[float, float]],
    vector_smoother: VectorSmoother,
    strength: float,
    max_shift_ratio: float,
    replace_mode: str,
    mask_scale: float,
    mask_blur: int,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if eye.iris is None or eye.ratio is None or calibrated_ratio is None or not eye.box.valid():
        return None

    copied = _copy_eye_patch(frame, eye.box, eye.eye_points, mask_scale, mask_blur)
    if copied is None:
        return None
    patch, mask = copied

    _, _, eye_w, eye_h = cv2.boundingRect(eye.eye_points)
    if eye_w < 10 or eye_h < 6:
        return None

    current_rx, current_ry = eye.ratio
    target_rx, target_ry = calibrated_ratio

    raw_dx = (target_rx - current_rx) * eye_w * strength
    raw_dy = (target_ry - current_ry) * eye_h * strength

    raw_dx = float(np.clip(raw_dx, -eye_w * max_shift_ratio, eye_w * max_shift_ratio))
    raw_dy = float(np.clip(raw_dy, -eye_h * max_shift_ratio, eye_h * max_shift_ratio))

    dx, dy = vector_smoother.update(eye_key, (raw_dx, raw_dy))

    if abs(dx) < 0.35 and abs(dy) < 0.35:
        start = (int(round(eye.iris[0])), int(round(eye.iris[1])))
        return start, start

    _fill_eye_region(frame, eye.box, eye.eye_points, mask_scale, replace_mode)
    _paste_patch(frame, patch, mask, eye.box, dx, dy)

    start = (int(round(eye.iris[0])), int(round(eye.iris[1])))
    end = (int(round(eye.iris[0] + dx)), int(round(eye.iris[1] + dy)))
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

    calibrated_left: Optional[Tuple[float, float]] = None
    calibrated_right: Optional[Tuple[float, float]] = None

    debug = args.debug
    correction_enabled = not args.no_correction
    virtual_enabled = args.virtual_camera
    strength = args.strength
    max_shift_ratio = args.max_shift_ratio
    current_fps = 0.0
    fps_count = 0
    fps_time = time.time()
    virtual_cam = None
    last_left_state: Optional[EyeState] = None
    last_right_state: Optional[EyeState] = None

    print(f"Using MediaPipe backend: {provider.backend}")
    print("Look directly at the camera and press k to calibrate.")

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
            correction_vectors = []
            ratio_text = "ratios=missing"

            if landmarks:
                left_eye = _extract_eye_state(landmarks, LEFT_EYE_OUTLINE, LEFT_IRIS, width, height, point_stabilizer, "left")
                right_eye = _extract_eye_state(landmarks, RIGHT_EYE_OUTLINE, RIGHT_IRIS, width, height, point_stabilizer, "right")
                last_left_state = left_eye
                last_right_state = right_eye
                ratio_text = f"L={left_eye.ratio} R={right_eye.ratio}"

                if correction_enabled and calibrated_left and calibrated_right:
                    left_vec = _replace_eye_to_calibrated_ratio(
                        frame,
                        "left_vec",
                        left_eye,
                        calibrated_left,
                        vector_smoother,
                        strength,
                        max_shift_ratio,
                        args.replace_mode,
                        args.mask_scale,
                        args.mask_blur,
                    )
                    right_vec = _replace_eye_to_calibrated_ratio(
                        frame,
                        "right_vec",
                        right_eye,
                        calibrated_right,
                        vector_smoother,
                        strength,
                        max_shift_ratio,
                        args.replace_mode,
                        args.mask_scale,
                        args.mask_blur,
                    )
                    correction_vectors = [v for v in (left_vec, right_vec) if v is not None]

                if debug:
                    _draw_debug(frame, landmarks, width, height)
                    cv2.rectangle(frame, (left_eye.box.x1, left_eye.box.y1), (left_eye.box.x2, left_eye.box.y2), (255, 255, 255), 1)
                    cv2.rectangle(frame, (right_eye.box.x1, right_eye.box.y1), (right_eye.box.x2, right_eye.box.y2), (255, 255, 255), 1)
                    for start, end in correction_vectors:
                        cv2.arrowedLine(frame, start, end, (0, 0, 255), 2, tipLength=0.35)

            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_time = time.time()

            calibrated = calibrated_left is not None and calibrated_right is not None
            _draw_hud(
                frame,
                [
                    f"v8 face={'detected' if face_detected else 'missing'} calibrated={'yes' if calibrated else 'no'} fps={current_fps:.1f}",
                    f"correction={'on' if correction_enabled else 'off'} strength={strength:.2f} max_shift={max_shift_ratio:.2f} mode={args.replace_mode}",
                    "look at camera then press k | c correction | d debug | +/- strength | [] shift",
                    ratio_text[:120],
                ],
                calibrated,
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

            cv2.imshow("Mac Eye Contact MVP V8", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("c"):
                correction_enabled = not correction_enabled
            if key == ord("k"):
                if last_left_state and last_right_state and last_left_state.ratio and last_right_state.ratio:
                    calibrated_left = last_left_state.ratio
                    calibrated_right = last_right_state.ratio
                    vector_smoother.state.clear()
                    print(f"Calibrated left={calibrated_left} right={calibrated_right}")
                else:
                    print("Could not calibrate: eye ratios missing")
            if key == ord("v"):
                virtual_enabled = not virtual_enabled
                if not virtual_enabled and virtual_cam is not None:
                    virtual_cam.close()
                    virtual_cam = None
            if key in (ord("+"), ord("=")):
                strength = min(2.0, strength + 0.05)
            if key in (ord("-"), ord("_")):
                strength = max(0.0, strength - 0.05)
            if key == ord("["):
                max_shift_ratio = max(0.02, max_shift_ratio - 0.02)
            if key == ord("]"):
                max_shift_ratio = min(0.35, max_shift_ratio + 0.02)
    finally:
        if virtual_cam is not None:
            virtual_cam.close()
        provider.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrated gaze correction macOS MVP V8")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--max-shift-ratio", type=float, default=0.16)
    parser.add_argument("--point-smoothing", type=float, default=0.88)
    parser.add_argument("--point-deadzone", type=float, default=1.0)
    parser.add_argument("--vector-smoothing", type=float, default=0.90)
    parser.add_argument("--vector-deadzone", type=float, default=0.7)
    parser.add_argument("--replace-mode", choices=["none", "patch", "inpaint"], default="patch")
    parser.add_argument("--mask-scale", type=float, default=1.18)
    parser.add_argument("--mask-blur", type=int, default=27)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--virtual-camera", action="store_true")
    parser.add_argument("--landmarker-model", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

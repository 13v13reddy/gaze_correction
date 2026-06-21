"""macOS Eye Contact MVP V4.

V4 adds manual gaze-target controls. Earlier versions targeted the iris toward
the geometric center of each eye, which often produces near-zero correction.
This version lets you tune the target offset live with keyboard controls.

Run:
    python mac_mvp/eye_contact_mvp_v4.py --debug --strength 1.0

Keys:
    q or Esc    quit
    d           toggle debug overlays
    c           toggle correction on/off
    w/s/a/d     move target up/down/left/right
    r           reset target offset
    + / -       increase/decrease correction strength

Typical use:
    1. Start with --debug.
    2. Look at the screen, not the camera.
    3. Use w/a/s/d until the red arrows point in the desired correction direction.
    4. Turn debug off with d and compare with c.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

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


def _draw_hud(frame, lines, face_detected: bool) -> None:
    bg_color = (30, 120, 30) if face_detected else (30, 30, 180)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 94), bg_color, -1)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def _eye_center(eye_points: np.ndarray) -> Tuple[float, float]:
    return float(np.median(eye_points[:, 0])), float(np.median(eye_points[:, 1]))


def _soft_mask(shape: Tuple[int, int], center: Tuple[float, float], axes: Tuple[float, float]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        (max(2, int(round(axes[0]))), max(2, int(round(axes[1])))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    feather = max(5, int(min(axes) * 0.8)) | 1
    return cv2.GaussianBlur(mask, (feather, feather), 0)[..., None]


def _redirect_eye_manual(
    frame: np.ndarray,
    box,
    eye_points: np.ndarray,
    iris_center: Optional[Tuple[float, float]],
    strength: float,
    target_offset_x: float,
    target_offset_y: float,
    max_shift_ratio: float,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if not box.valid() or iris_center is None:
        return None

    roi = frame[box.y1:box.y2, box.x1:box.x2]
    if roi.size == 0:
        return None

    h, w = roi.shape[:2]
    iris_x = iris_center[0] - box.x1
    iris_y = iris_center[1] - box.y1

    eye_cx, eye_cy = _eye_center(eye_points)
    target_x = (eye_cx - box.x1) + target_offset_x * w
    target_y = (eye_cy - box.y1) + target_offset_y * h

    dx = (target_x - iris_x) * strength
    dy = (target_y - iris_y) * strength

    dx = float(np.clip(dx, -w * max_shift_ratio, w * max_shift_ratio))
    dy = float(np.clip(dy, -h * max_shift_ratio, h * max_shift_ratio))

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(roi, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    _, _, eye_w, eye_h = cv2.boundingRect(eye_points)
    eye_w = max(10.0, float(eye_w))
    eye_h = max(6.0, float(eye_h))

    # Wider mask than V2/V3 because manual testing needs a visible effect.
    mask = _soft_mask(
        (h, w),
        center=(iris_x + dx * 0.2, iris_y + dy * 0.2),
        axes=(eye_w * 0.38, eye_h * 1.05),
    )

    blended = (shifted.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
    frame[box.y1:box.y2, box.x1:box.x2] = blended

    start = (int(iris_x + box.x1), int(iris_y + box.y1))
    end = (int(iris_x + dx + box.x1), int(iris_y + dy + box.y1))
    return start, end


def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS camera permissions and camera index.")

    provider = LandmarkProvider(args.landmarker_model)
    stabilizer = Stabilizer(alpha=args.smoothing, deadzone=args.deadzone)
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
                left_box = _bounding_box(left_eye_points, width, height)
                right_box = _bounding_box(right_eye_points, width, height)

                left_iris = _iris_center(landmarks, LEFT_IRIS, width, height)
                right_iris = _iris_center(landmarks, RIGHT_IRIS, width, height)
                iris_status = f"iris=left:{left_iris is not None} right:{right_iris is not None}"

                if left_iris is not None:
                    left_iris = stabilizer.update("left_iris", left_iris)
                if right_iris is not None:
                    right_iris = stabilizer.update("right_iris", right_iris)

                if correction_enabled:
                    left_vec = _redirect_eye_manual(
                        frame,
                        left_box,
                        left_eye_points,
                        left_iris,
                        strength,
                        target_offset_x,
                        target_offset_y,
                        args.max_shift_ratio,
                    )
                    right_vec = _redirect_eye_manual(
                        frame,
                        right_box,
                        right_eye_points,
                        right_iris,
                        strength,
                        target_offset_x,
                        target_offset_y,
                        args.max_shift_ratio,
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
                    f"v4 face={'detected' if face_detected else 'missing'} {iris_status} fps={current_fps:.1f}",
                    f"correction={'on' if correction_enabled else 'off'} strength={strength:.2f} target=({target_offset_x:+.2f},{target_offset_y:+.2f})",
                    "keys: w/s/a/d move target | c toggle correction | d debug | r reset",
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

            cv2.imshow("Mac Eye Contact MVP V4", frame)
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
                target_offset_y -= 0.03
            if key == ord("s"):
                target_offset_y += 0.03
            if key == ord("a"):
                target_offset_x -= 0.03
            if key == ord("d"):
                # d is already debug toggle. Use Shift+D is not portable with cv2.waitKey,
                # so right movement is also mapped to l.
                pass
            if key == ord("l"):
                target_offset_x += 0.03
            if key in (ord("+"), ord("=")):
                strength = min(2.0, strength + 0.10)
            if key in (ord("-"), ord("_")):
                strength = max(0.0, strength - 0.10)
    finally:
        if virtual_cam is not None:
            virtual_cam.close()
        provider.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually tunable macOS eye-contact MVP V4")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.75)
    parser.add_argument("--deadzone", type=float, default=0.8)
    parser.add_argument("--max-shift-ratio", type=float, default=0.28)
    parser.add_argument("--target-offset-x", type=float, default=0.0)
    parser.add_argument("--target-offset-y", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--virtual-camera", action="store_true")
    parser.add_argument("--landmarker-model", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""macOS Eye Contact MVP V5.

V5 replaces whole-eye ROI warping with an iris-clone approach:

1. Track iris landmarks.
2. Copy a small patch around the iris.
3. Inpaint/soft-fill the old iris zone.
4. Paste the iris patch at a corrected target position.

This is still not neural eye synthesis, but it avoids the heavy ghosting caused
by shifting the entire eye texture in V4.

Run:
    python mac_mvp/eye_contact_mvp_v5.py --debug --strength 1.0

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


def _eye_center(eye_points: np.ndarray) -> Tuple[float, float]:
    return float(np.median(eye_points[:, 0])), float(np.median(eye_points[:, 1]))


def _draw_hud(frame, lines, face_detected: bool) -> None:
    bg_color = (30, 120, 30) if face_detected else (30, 30, 180)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 96), bg_color, -1)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def _ellipse_mask(shape: Tuple[int, int], center: Tuple[float, float], axes: Tuple[float, float], blur: int = 9) -> np.ndarray:
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
    blur = max(3, blur | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]


def _copy_patch_with_mask(image: np.ndarray, center: Tuple[float, float], radius_x: int, radius_y: int):
    h, w = image.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1 = max(0, cx - radius_x)
    y1 = max(0, cy - radius_y)
    x2 = min(w, cx + radius_x)
    y2 = min(h, cy + radius_y)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2].copy()
    local_center = (cx - x1, cy - y1)
    mask = _ellipse_mask((patch.shape[0], patch.shape[1]), local_center, (radius_x * 0.82, radius_y * 0.82), blur=7)
    return patch, mask, (x1, y1, x2, y2)


def _blend_patch(image: np.ndarray, patch: np.ndarray, mask: np.ndarray, center: Tuple[float, float]) -> None:
    h, w = image.shape[:2]
    ph, pw = patch.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1 = cx - pw // 2
    y1 = cy - ph // 2
    x2 = x1 + pw
    y2 = y1 + ph

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(w, x2)
    dst_y2 = min(h, y2)
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return

    roi = image[dst_y1:dst_y2, dst_x1:dst_x2]
    src_patch = patch[src_y1:src_y2, src_x1:src_x2]
    src_mask = mask[src_y1:src_y2, src_x1:src_x2]
    image[dst_y1:dst_y2, dst_x1:dst_x2] = (src_patch.astype(np.float32) * src_mask + roi.astype(np.float32) * (1.0 - src_mask)).astype(np.uint8)


def _fill_old_iris(frame: np.ndarray, center: Tuple[float, float], radius_x: int, radius_y: int) -> None:
    h, w = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (max(2, radius_x), max(2, radius_y)), 0, 0, 360, 255, -1)
    # Inpaint only a small zone. This is imperfect but prevents double-iris ghosting.
    filled = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (9, 9), 0)[..., None]
    frame[:] = (filled.astype(np.float32) * soft + frame.astype(np.float32) * (1.0 - soft)).astype(np.uint8)


def _redirect_eye_clone(
    frame: np.ndarray,
    eye_points: np.ndarray,
    iris_center: Optional[Tuple[float, float]],
    strength: float,
    target_offset_x: float,
    target_offset_y: float,
    max_shift_ratio: float,
    erase_old_iris: bool,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if iris_center is None:
        return None

    eye_x, eye_y, eye_w, eye_h = cv2.boundingRect(eye_points)
    if eye_w < 10 or eye_h < 6:
        return None

    eye_cx, eye_cy = _eye_center(eye_points)
    target_x = eye_cx + target_offset_x * eye_w
    target_y = eye_cy + target_offset_y * eye_h

    raw_dx = (target_x - iris_center[0]) * strength
    raw_dy = (target_y - iris_center[1]) * strength
    max_dx = eye_w * max_shift_ratio
    max_dy = eye_h * max_shift_ratio
    dx = float(np.clip(raw_dx, -max_dx, max_dx))
    dy = float(np.clip(raw_dy, -max_dy, max_dy))

    new_center = (iris_center[0] + dx, iris_center[1] + dy)

    # The patch must include iris and a bit of sclera/eyelid context.
    radius_x = max(5, int(eye_w * 0.24))
    radius_y = max(4, int(eye_h * 0.62))
    copied = _copy_patch_with_mask(frame, iris_center, radius_x, radius_y)
    if copied is None:
        return None
    patch, mask, _ = copied

    if erase_old_iris and (abs(dx) > 0.5 or abs(dy) > 0.5):
        _fill_old_iris(frame, iris_center, max(3, int(radius_x * 0.62)), max(3, int(radius_y * 0.55)))

    _blend_patch(frame, patch, mask, new_center)

    start = (int(round(iris_center[0])), int(round(iris_center[1])))
    end = (int(round(new_center[0])), int(round(new_center[1])))
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
                    left_vec = _redirect_eye_clone(
                        frame,
                        left_eye_points,
                        left_iris,
                        strength,
                        target_offset_x,
                        target_offset_y,
                        args.max_shift_ratio,
                        args.erase_old_iris,
                    )
                    right_vec = _redirect_eye_clone(
                        frame,
                        right_eye_points,
                        right_iris,
                        strength,
                        target_offset_x,
                        target_offset_y,
                        args.max_shift_ratio,
                        args.erase_old_iris,
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
                    f"v5 face={'detected' if face_detected else 'missing'} {iris_status} fps={current_fps:.1f}",
                    f"correction={'on' if correction_enabled else 'off'} strength={strength:.2f} target=({target_offset_x:+.2f},{target_offset_y:+.2f}) erase={args.erase_old_iris}",
                    "keys: w/s/a/l target | c correction | d debug | r reset | +/- strength",
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

            cv2.imshow("Mac Eye Contact MVP V5", frame)
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
                target_offset_y -= 0.04
            if key == ord("s"):
                target_offset_y += 0.04
            if key == ord("a"):
                target_offset_x -= 0.04
            if key == ord("l"):
                target_offset_x += 0.04
            if key in (ord("+"), ord("=")):
                strength = min(2.5, strength + 0.10)
            if key in (ord("-"), ord("_")):
                strength = max(0.0, strength - 0.10)
    finally:
        if virtual_cam is not None:
            virtual_cam.close()
        provider.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iris-clone macOS eye-contact MVP V5")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.78)
    parser.add_argument("--deadzone", type=float, default=0.6)
    parser.add_argument("--max-shift-ratio", type=float, default=0.38)
    parser.add_argument("--target-offset-x", type=float, default=0.0)
    parser.add_argument("--target-offset-y", type=float, default=0.0)
    parser.add_argument("--erase-old-iris", action="store_true", help="try to remove the old iris location before pasting the corrected iris")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--virtual-camera", action="store_true")
    parser.add_argument("--landmarker-model", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""macOS Eye Contact MVP V6 - full-eye replacement experiment.

V6 replaces the whole eye patch instead of only the iris. It copies an eye
region, optionally fills the original region, and pastes the shifted eye patch
back using a feathered mask based on the eye landmarks.

This is still classical OpenCV rendering, not neural synthesis. Expect artifacts,
but this version helps compare full-eye replacement against iris-only cloning.

Run:
    python mac_mvp/eye_contact_mvp_v6.py --debug --strength 1.0

Useful tests:
    python mac_mvp/eye_contact_mvp_v6.py --debug --strength 1.2 --replace-mode patch
    python mac_mvp/eye_contact_mvp_v6.py --debug --strength 1.2 --replace-mode inpaint

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
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 104), bg_color, -1)
    y = 30
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def _soft_eye_mask(shape: Tuple[int, int], eye_points_local: np.ndarray, scale: float, blur: int) -> np.ndarray:
    h, w = shape
    center = eye_points_local.mean(axis=0)
    expanded = (eye_points_local.astype(np.float32) - center) * scale + center
    expanded = expanded.astype(np.int32)

    mask = np.zeros((h, w), dtype=np.float32)
    hull = cv2.convexHull(expanded)
    cv2.fillConvexPoly(mask, hull, 1.0)

    blur = max(3, blur | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask[..., None]


def _copy_eye_patch(frame: np.ndarray, box, eye_points: np.ndarray, mask_scale: float, mask_blur: int):
    if not box.valid():
        return None

    patch = frame[box.y1:box.y2, box.x1:box.x2].copy()
    if patch.size == 0:
        return None

    local_points = eye_points.copy().astype(np.int32)
    local_points[:, 0] -= box.x1
    local_points[:, 1] -= box.y1
    mask = _soft_eye_mask((patch.shape[0], patch.shape[1]), local_points, mask_scale, mask_blur)
    return patch, mask


def _paste_patch(frame: np.ndarray, patch: np.ndarray, mask: np.ndarray, box, dx: float, dy: float) -> None:
    h, w = frame.shape[:2]
    ph, pw = patch.shape[:2]

    x1 = int(round(box.x1 + dx))
    y1 = int(round(box.y1 + dy))
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

    roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
    src_patch = patch[src_y1:src_y2, src_x1:src_x2]
    src_mask = mask[src_y1:src_y2, src_x1:src_x2]
    frame[dst_y1:dst_y2, dst_x1:dst_x2] = (src_patch.astype(np.float32) * src_mask + roi.astype(np.float32) * (1.0 - src_mask)).astype(np.uint8)


def _fill_eye_region(frame: np.ndarray, box, eye_points: np.ndarray, mask_scale: float, mode: str) -> None:
    if mode == "none":
        return

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = eye_points.mean(axis=0)
    expanded = (eye_points.astype(np.float32) - center) * mask_scale + center
    hull = cv2.convexHull(expanded.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    if mode == "inpaint":
        filled = cv2.inpaint(frame, mask, 4, cv2.INPAINT_TELEA)
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (13, 13), 0)[..., None]
        frame[:] = (filled.astype(np.float32) * soft + frame.astype(np.float32) * (1.0 - soft)).astype(np.uint8)
        return

    if mode == "patch":
        # Simple local blur fill. Less destructive than inpaint, but may leave a ghost.
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (13, 13), 0)[..., None]
        frame[:] = (blurred.astype(np.float32) * soft + frame.astype(np.float32) * (1.0 - soft)).astype(np.uint8)


def _replace_whole_eye(
    frame: np.ndarray,
    box,
    eye_points: np.ndarray,
    iris_center: Optional[Tuple[float, float]],
    strength: float,
    target_offset_x: float,
    target_offset_y: float,
    max_shift_ratio: float,
    replace_mode: str,
    mask_scale: float,
    mask_blur: int,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if iris_center is None or not box.valid():
        return None

    copied = _copy_eye_patch(frame, box, eye_points, mask_scale, mask_blur)
    if copied is None:
        return None
    patch, mask = copied

    eye_x, eye_y, eye_w, eye_h = cv2.boundingRect(eye_points)
    eye_cx, eye_cy = _eye_center(eye_points)

    target_x = eye_cx + target_offset_x * eye_w
    target_y = eye_cy + target_offset_y * eye_h

    dx = (target_x - iris_center[0]) * strength
    dy = (target_y - iris_center[1]) * strength

    dx = float(np.clip(dx, -eye_w * max_shift_ratio, eye_w * max_shift_ratio))
    dy = float(np.clip(dy, -eye_h * max_shift_ratio, eye_h * max_shift_ratio))

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
                left_box = _bounding_box(left_eye_points, width, height, pad_ratio=args.box_padding)
                right_box = _bounding_box(right_eye_points, width, height, pad_ratio=args.box_padding)

                left_iris = _iris_center(landmarks, LEFT_IRIS, width, height)
                right_iris = _iris_center(landmarks, RIGHT_IRIS, width, height)
                iris_status = f"iris=left:{left_iris is not None} right:{right_iris is not None}"

                if left_iris is not None:
                    left_iris = stabilizer.update("left_iris", left_iris)
                if right_iris is not None:
                    right_iris = stabilizer.update("right_iris", right_iris)

                if correction_enabled:
                    left_vec = _replace_whole_eye(
                        frame,
                        left_box,
                        left_eye_points,
                        left_iris,
                        strength,
                        target_offset_x,
                        target_offset_y,
                        args.max_shift_ratio,
                        args.replace_mode,
                        args.mask_scale,
                        args.mask_blur,
                    )
                    right_vec = _replace_whole_eye(
                        frame,
                        right_box,
                        right_eye_points,
                        right_iris,
                        strength,
                        target_offset_x,
                        target_offset_y,
                        args.max_shift_ratio,
                        args.replace_mode,
                        args.mask_scale,
                        args.mask_blur,
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
                    f"v6 face={'detected' if face_detected else 'missing'} {iris_status} fps={current_fps:.1f}",
                    f"whole-eye replace={args.replace_mode} correction={'on' if correction_enabled else 'off'} strength={strength:.2f}",
                    f"target=({target_offset_x:+.2f},{target_offset_y:+.2f}) keys: w/s/a/l target | c correction | d debug | r reset",
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

            cv2.imshow("Mac Eye Contact MVP V6", frame)
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
    parser = argparse.ArgumentParser(description="Full-eye replacement macOS eye-contact MVP V6")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.78)
    parser.add_argument("--deadzone", type=float, default=0.6)
    parser.add_argument("--max-shift-ratio", type=float, default=0.32)
    parser.add_argument("--target-offset-x", type=float, default=0.0)
    parser.add_argument("--target-offset-y", type=float, default=0.0)
    parser.add_argument("--replace-mode", choices=["none", "patch", "inpaint"], default="patch")
    parser.add_argument("--box-padding", type=float, default=0.45)
    parser.add_argument("--mask-scale", type=float, default=1.55)
    parser.add_argument("--mask-blur", type=int, default=13)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--virtual-camera", action="store_true")
    parser.add_argument("--landmarker-model", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

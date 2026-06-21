"""macOS Eye Contact MVP V2.

This version improves the first MVP by adding temporal smoothing and a
stabilized iris-target correction. It is still lightweight and does not use a
neural synthesis model yet.

Run:
    python mac_mvp/eye_contact_mvp_v2.py --debug

Useful tuning:
    python mac_mvp/eye_contact_mvp_v2.py --strength 0.65 --smoothing 0.82 --deadzone 1.8

Keys:
    q or Esc  quit
    d         toggle debug overlays
    v         toggle virtual camera output
    + / -     increase/decrease correction strength
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None

try:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_tasks_vision
except ImportError:
    mp_tasks_python = None
    mp_tasks_vision = None

LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")


@dataclass
class EyeBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def valid(self) -> bool:
        return self.width > 10 and self.height > 8


class Stabilizer:
    """Small exponential moving average for jitter reduction."""

    def __init__(self, alpha: float, deadzone: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 0.98))
        self.deadzone = max(0.0, deadzone)
        self.state: Dict[str, np.ndarray] = {}

    def update(self, key: str, point: Tuple[float, float]) -> Tuple[float, float]:
        value = np.asarray(point, dtype=np.float32)
        previous = self.state.get(key)
        if previous is None:
            self.state[key] = value
            return float(value[0]), float(value[1])

        if np.linalg.norm(value - previous) < self.deadzone:
            return float(previous[0]), float(previous[1])

        smoothed = self.alpha * previous + (1.0 - self.alpha) * value
        self.state[key] = smoothed
        return float(smoothed[0]), float(smoothed[1])


def _points_from_landmarks(landmarks, indices: Iterable[int], width: int, height: int) -> np.ndarray:
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append([int(lm.x * width), int(lm.y * height)])
    return np.asarray(points, dtype=np.int32)


def _bounding_box(points: np.ndarray, width: int, height: int, pad_ratio: float = 0.38) -> EyeBox:
    x, y, w, h = cv2.boundingRect(points)
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    return EyeBox(
        x1=max(0, x - pad_x),
        y1=max(0, y - pad_y),
        x2=min(width, x + w + pad_x),
        y2=min(height, y + h + pad_y),
    )


def _iris_center(landmarks, iris_indices: Iterable[int], width: int, height: int) -> Optional[Tuple[float, float]]:
    if len(landmarks) <= max(iris_indices):
        return None
    points = _points_from_landmarks(landmarks, iris_indices, width, height)
    if points.size == 0:
        return None
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def _eye_center(eye_points: np.ndarray) -> Tuple[float, float]:
    return float(np.median(eye_points[:, 0])), float(np.median(eye_points[:, 1]))


def _soft_ellipse_mask(shape: Tuple[int, int], center: Tuple[float, float], axes: Tuple[float, float]) -> np.ndarray:
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
    feather = max(5, int(min(axes) * 0.75)) | 1
    return cv2.GaussianBlur(mask, (feather, feather), 0)[..., None]


def _redirect_eye_stable(
    frame: np.ndarray,
    box: EyeBox,
    eye_points: np.ndarray,
    iris_center: Optional[Tuple[float, float]],
    strength: float,
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
    target_x, target_y = _eye_center(eye_points)
    target_x -= box.x1
    target_y -= box.y1

    raw_dx = (target_x - iris_x) * strength
    raw_dy = (target_y - iris_y) * strength * 0.55

    dx = float(np.clip(raw_dx, -w * max_shift_ratio, w * max_shift_ratio))
    dy = float(np.clip(raw_dy, -h * max_shift_ratio * 0.65, h * max_shift_ratio * 0.65))

    if abs(dx) < 0.25 and abs(dy) < 0.25:
        return ((int(iris_x + box.x1), int(iris_y + box.y1)), (int(iris_x + box.x1), int(iris_y + box.y1)))

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(roi, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    _, _, eye_w, eye_h = cv2.boundingRect(eye_points)
    eye_w = max(10.0, float(eye_w))
    eye_h = max(6.0, float(eye_h))

    # Restrict the mask to the iris/pupil zone. This reduces the monster-eye
    # look and preserves eyelids better than shifting the whole eye box.
    mask = _soft_ellipse_mask(
        (h, w),
        center=(iris_x + dx * 0.10, iris_y + dy * 0.10),
        axes=(eye_w * 0.26, eye_h * 0.82),
    )

    # Slight blur before blending to reduce shimmer from pixel-level jitter.
    shifted = cv2.GaussianBlur(shifted, (3, 3), 0)
    blended = (shifted.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
    frame[box.y1:box.y2, box.x1:box.x2] = blended

    start = (int(iris_x + box.x1), int(iris_y + box.y1))
    end = (int(iris_x + dx + box.x1), int(iris_y + dy + box.y1))
    return start, end


def _draw_debug(frame: np.ndarray, landmarks, width: int, height: int) -> None:
    for indices in (LEFT_EYE_OUTLINE, RIGHT_EYE_OUTLINE, LEFT_IRIS, RIGHT_IRIS):
        if len(landmarks) <= max(indices):
            continue
        points = _points_from_landmarks(landmarks, indices, width, height)
        for x, y in points:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


class LandmarkProvider:
    def __init__(self, model_path: str) -> None:
        self.backend = None
        self.face_mesh = None
        self.face_landmarker = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.backend = "solutions"
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        if mp_tasks_python is None or mp_tasks_vision is None:
            raise RuntimeError("MediaPipe Tasks API could not be imported.")

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"MediaPipe FaceLandmarker model not found: {model_path}\n"
                "Run: python mac_mvp/download_models.py"
            )

        self.backend = "tasks"
        base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
        options = mp_tasks_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.face_landmarker is not None:
            self.face_landmarker.close()

    def process(self, rgb_frame: np.ndarray):
        if self.backend == "solutions":
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.face_landmarker.detect(image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]


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
    virtual_enabled = args.virtual_camera
    strength = args.strength
    fps_time = time.time()
    fps_count = 0
    current_fps = 0.0
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
            correction_vectors = []

            if landmarks:
                left_eye_points = _points_from_landmarks(landmarks, LEFT_EYE_OUTLINE, width, height)
                right_eye_points = _points_from_landmarks(landmarks, RIGHT_EYE_OUTLINE, width, height)
                left_box = _bounding_box(left_eye_points, width, height)
                right_box = _bounding_box(right_eye_points, width, height)

                left_iris = _iris_center(landmarks, LEFT_IRIS, width, height)
                right_iris = _iris_center(landmarks, RIGHT_IRIS, width, height)

                if left_iris is not None:
                    left_iris = stabilizer.update("left_iris", left_iris)
                if right_iris is not None:
                    right_iris = stabilizer.update("right_iris", right_iris)

                left_vec = _redirect_eye_stable(frame, left_box, left_eye_points, left_iris, strength, args.max_shift_ratio)
                right_vec = _redirect_eye_stable(frame, right_box, right_eye_points, right_iris, strength, args.max_shift_ratio)
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
                fps_time = time.time()
                fps_count = 0

            cv2.putText(
                frame,
                f"v2 strength={strength:.2f} smooth={args.smoothing:.2f} fps={current_fps:.1f} virtual={'on' if virtual_enabled else 'off'}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
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

            cv2.imshow("Mac Eye Contact MVP V2", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("v"):
                virtual_enabled = not virtual_enabled
                if not virtual_enabled and virtual_cam is not None:
                    virtual_cam.close()
                    virtual_cam = None
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
    parser = argparse.ArgumentParser(description="Stabilized macOS eye-contact MVP V2")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--smoothing", type=float, default=0.82, help="EMA smoothing. Higher means less jitter but more lag.")
    parser.add_argument("--deadzone", type=float, default=1.8, help="Ignore tiny iris movements under this many pixels.")
    parser.add_argument("--max-shift-ratio", type=float, default=0.18, help="Clamp max iris shift relative to eye ROI size.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--virtual-camera", action="store_true")
    parser.add_argument("--landmarker-model", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

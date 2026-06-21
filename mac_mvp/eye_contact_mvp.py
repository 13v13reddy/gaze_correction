"""macOS eye-contact MVP prototype.

This is a lightweight proof of concept for simulating eye contact on macOS.
It supports both MediaPipe APIs:

1. Older `mediapipe.solutions.face_mesh` API.
2. Newer Python 3.12 `mediapipe.tasks` FaceLandmarker API.

Run:
    python mac_mvp/eye_contact_mvp.py --debug

If your MediaPipe package only exposes `tasks`, download the model first:
    python mac_mvp/download_models.py

Keys:
    q or Esc  quit
    d         toggle debug landmarks
    v         toggle virtual camera output if pyvirtualcam is installed
    + / -     increase/decrease correction strength
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyvirtualcam
except ImportError:  # optional dependency
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
        return self.width > 8 and self.height > 8


def _points_from_landmarks(landmarks, indices: Iterable[int], width: int, height: int) -> np.ndarray:
    points = []
    for idx in indices:
        lm = landmarks[idx]
        points.append([int(lm.x * width), int(lm.y * height)])
    return np.asarray(points, dtype=np.int32)


def _bounding_box(points: np.ndarray, width: int, height: int, pad_ratio: float = 0.55) -> EyeBox:
    x, y, w, h = cv2.boundingRect(points)
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    return EyeBox(
        x1=max(0, x - pad_x),
        y1=max(0, y - pad_y),
        x2=min(width, x + w + pad_x),
        y2=min(height, y + h + pad_y),
    )


def _iris_center(landmarks, iris_indices: Iterable[int], width: int, height: int) -> Optional[Tuple[int, int]]:
    if len(landmarks) <= max(iris_indices):
        return None
    points = _points_from_landmarks(landmarks, iris_indices, width, height)
    if points.size == 0:
        return None
    center = points.mean(axis=0).astype(int)
    return int(center[0]), int(center[1])


def _soft_mask(shape: Tuple[int, int], feather: int = 12) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 1.0, -1)
    k = max(3, feather | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask[..., None]


def _redirect_eye(frame: np.ndarray, box: EyeBox, iris_center: Optional[Tuple[int, int]], strength: float) -> None:
    if not box.valid() or iris_center is None:
        return

    roi = frame[box.y1:box.y2, box.x1:box.x2]
    if roi.size == 0:
        return

    h, w = roi.shape[:2]
    local_iris_x = iris_center[0] - box.x1
    local_iris_y = iris_center[1] - box.y1

    target_x = w * 0.5
    target_y = h * 0.5

    dx = (target_x - local_iris_x) * strength
    dy = (target_y - local_iris_y) * strength * 0.35

    dx = float(np.clip(dx, -w * 0.10, w * 0.10))
    dy = float(np.clip(dy, -h * 0.06, h * 0.06))

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(roi, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mask = _soft_mask((h, w), feather=max(7, min(w, h) // 3))
    blended = (shifted.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
    frame[box.y1:box.y2, box.x1:box.x2] = blended


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
            raise RuntimeError(
                "This MediaPipe install does not expose mp.solutions and the tasks API could not be imported. "
                "Try: pip install --upgrade 'mediapipe>=0.10,<0.11'"
            )

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"MediaPipe FaceLandmarker model not found: {model_path}\n"
                "Run: python mac_mvp/download_models.py\n"
                "Then run this script again."
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

            if landmarks:
                left_eye_points = _points_from_landmarks(landmarks, LEFT_EYE_OUTLINE, width, height)
                right_eye_points = _points_from_landmarks(landmarks, RIGHT_EYE_OUTLINE, width, height)
                left_box = _bounding_box(left_eye_points, width, height)
                right_box = _bounding_box(right_eye_points, width, height)
                left_iris = _iris_center(landmarks, LEFT_IRIS, width, height)
                right_iris = _iris_center(landmarks, RIGHT_IRIS, width, height)

                _redirect_eye(frame, left_box, left_iris, strength)
                _redirect_eye(frame, right_box, right_iris, strength)

                if debug:
                    _draw_debug(frame, landmarks, width, height)
                    cv2.rectangle(frame, (left_box.x1, left_box.y1), (left_box.x2, left_box.y2), (255, 255, 255), 1)
                    cv2.rectangle(frame, (right_box.x1, right_box.y1), (right_box.x2, right_box.y2), (255, 255, 255), 1)

            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_time = time.time()
                fps_count = 0

            cv2.putText(
                frame,
                f"strength={strength:.2f} fps={current_fps:.1f} virtual={'on' if virtual_enabled else 'off'}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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

            cv2.imshow("Mac Eye Contact MVP", frame)
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
                strength = min(1.0, strength + 0.05)
            if key in (ord("-"), ord("_")):
                strength = max(0.0, strength - 0.05)
    finally:
        if virtual_cam is not None:
            virtual_cam.close()
        provider.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="macOS eye-contact MVP using MediaPipe and OpenCV")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--fps", type=int, default=30, help="target FPS")
    parser.add_argument("--strength", type=float, default=0.35, help="correction strength from 0.0 to 1.0")
    parser.add_argument("--debug", action="store_true", help="show landmarks and eye boxes")
    parser.add_argument("--virtual-camera", action="store_true", help="send frames to pyvirtualcam when available")
    parser.add_argument("--landmarker-model", default=DEFAULT_MODEL_PATH, help="path to face_landmarker.task")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

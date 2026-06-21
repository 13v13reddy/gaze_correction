"""Set up model files required by the macOS eye-contact MVP.

Run:
    python mac_mvp/download_models.py

This helper creates mac_mvp/models and prints the command needed to download
MediaPipe's Face Landmarker model into the expected location.
"""

from __future__ import annotations

import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        print(f"Model is already present: {MODEL_PATH}")
        print("Next step:")
        print("  python mac_mvp/eye_contact_mvp.py --debug")
        return 0

    print("MediaPipe Face Landmarker model is missing.")
    print("Run this command from the repository root:")
    print()
    print(f"curl -L -o {MODEL_PATH} {MODEL_URL}")
    print()
    print("Then run:")
    print("  python mac_mvp/eye_contact_mvp.py --debug")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

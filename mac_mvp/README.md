# macOS Eye Contact MVP

This branch contains a lightweight eye-contact correction prototype for macOS.

## Features

- MediaPipe Face Mesh eye tracking
- Simple eye-region gaze redirection
- Real-time webcam preview
- Optional virtual camera output via pyvirtualcam
- Apple Silicon friendly

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r mac_mvp/requirements.txt
```

## Run

```bash
python mac_mvp/eye_contact_mvp.py --debug
```

## Controls

- q / Esc : quit
- d : toggle debug overlays
- v : toggle virtual camera
- + : increase correction strength
- - : decrease correction strength

## Notes

This is not NVIDIA Broadcast quality yet.
It is a foundation for future improvements:

- gaze estimation model
- CoreML acceleration
- temporal smoothing
- blink preservation
- neural eye synthesis

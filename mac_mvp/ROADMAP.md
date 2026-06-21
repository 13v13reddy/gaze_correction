# macOS Eye Contact MVP Roadmap

## Current status

The fast MVP proved the webcam and landmark pipeline works on macOS:

- Camera capture works.
- MediaPipe Face Landmarker works with Python 3.12 through the Tasks API.
- Face detection works.
- Eye landmarks work.
- Iris landmarks work.
- Frame rate is usable for experimentation.

## What failed

The OpenCV-only correction approach is not good enough.

Versions tested:

- `eye_contact_mvp.py`: whole-eye ROI warp. Too subtle, visible jitter.
- `eye_contact_mvp_v2.py`: smoothed iris-target warp. More stable but weak.
- `eye_contact_mvp_v3.py`: status/debug build. Confirmed tracking works.
- `eye_contact_mvp_v4.py`: manual target warp. Strong effect but heavy ghosting/smearing.
- `eye_contact_mvp_v5.py`: iris clone/inpaint attempt. Effect is visible but unnatural and cartoon-like.

Conclusion: the blocker is not landmark tracking. The blocker is image synthesis/rendering.

## Why the classical approach fails

The current pipeline tries to move pixels inside the eye region. This causes:

- duplicated irises
- smeared eyelids
- unstable eye texture
- unnatural pupil placement
- bad results when the head moves or the eyelids change shape

This is expected. A reliable eye-contact effect needs gaze estimation and learned eye rendering, not just pixel warping.

## Recommended next architecture

```text
Camera frame
  -> Face + iris landmarks
  -> Head pose estimation
  -> Gaze vector estimation
  -> Desired camera-facing gaze vector
  -> Eye-region synthesis model
  -> Temporal smoothing
  -> Blend corrected eye region back into frame
  -> Virtual camera output
```

## Phase 2 deliverables

1. Add head pose estimation using MediaPipe landmarks and OpenCV `solvePnP`.
2. Estimate approximate gaze direction from iris location and head pose.
3. Produce debug overlays showing:
   - face box
   - eye boxes
   - head pose axes
   - estimated gaze vector
   - desired camera-facing vector
4. Disable all destructive eye pixel manipulation by default.
5. Keep V1-V5 as experiments, but do not treat them as production candidates.

## Phase 3 deliverables

1. Evaluate a learned gaze redirection / eye synthesis model.
2. Convert the selected model to ONNX or Core ML.
3. Add temporal smoothing across generated eye regions.
4. Add OBS Virtual Camera or macOS camera extension path.

## Recommendation

Stop investing in classical OpenCV warping for the final product.

Use the current branch as proof that tracking works, then move to a model-based renderer.

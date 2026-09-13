# classsense/mp_pool.py
# A pool of MediaPipe FaceLandmarker instances plus the crop preparation that
# feeds them.
#
# A FaceLandmarker is stateful and not safe to share across threads, so each
# worker checks one out, uses it, and returns it. detect() releases the GIL -
# measured at 3.21x across 8 threads on 20 cores - which is what makes 60
# students per cycle affordable on a CPU-only machine.

import os
import queue
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from classsense.config import MP_MODEL, MP_MODEL_URL, MP_CROP_MAX_SIDE


def ensure_model(path=MP_MODEL, url=MP_MODEL_URL):
    """Download the landmarker weights on first run."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading MediaPipe face landmarker to {path} ...", flush=True)
    urllib.request.urlretrieve(url, path)
    print("Done.", flush=True)
    return path


def prepare_crop(frame_bgr, box, padding, max_side=MP_CROP_MAX_SIDE):
    """
    Cut a padded crop out of the frame and size it for MediaPipe.

    Returns (mp_image, width, height, scale) or None when the box is
    degenerate. `scale` is resized/original, so a caller can convert a
    measurement taken in the resized crop back into real camera pixels -
    which is what resolution tiering has to be judged on. Without it every
    student would be tiered by their post-downscale size and the whole room
    would read COARSE.

    Two things matter here:

    1. Aspect ratio is preserved. EAR and MAR are ratios of a vertical distance
       to a horizontal one, so they only survive a resize if both axes scale by
       the same factor. Squashing the crop to a square would quietly bias every
       eye and mouth measurement.

    2. The long side is capped, never stretched. Measured cost is flat from 96px
       to 192px (7.13 / 7.21 / 6.71 ms) and doubles at 256px (14.67 ms), so
       shrinking a large front-row crop is close to a free 2x. Upscaling a small
       crop, by contrast, adds no information and only costs time - a 40px face
       stays a 40px face however many pixels it is printed on.
    """
    h_frame, w_frame = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w_frame, x2 + padding)
    y2 = min(h_frame, y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    longest = max(ch, cw)
    scale = 1.0
    if longest > max_side:
        scale = max_side / float(longest)
        crop = cv2.resize(
            crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
            interpolation=cv2.INTER_AREA,
        )
        ch, cw = crop.shape[:2]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=np.ascontiguousarray(rgb))
    return image, cw, ch, scale


class MediaPipePool:
    """
    Fixed pool of landmarker instances, handed out one per concurrent caller.

    A plain Queue does the checkout: get() blocks until a detector is free, so
    the queue is both the free-list and the concurrency limit, and there is no
    separate semaphore to keep in step with it.
    """

    def __init__(self, pool_size, model_path=None, face_conf=0.45):
        model_path = ensure_model(model_path or MP_MODEL)
        self._free = queue.Queue()
        self._all = []

        for _ in range(pool_size):
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                num_faces=1,
                min_face_detection_confidence=face_conf,
                min_face_presence_confidence=face_conf,
                min_tracking_confidence=face_conf,
            )
            detector = mp_vision.FaceLandmarker.create_from_options(options)
            self._all.append(detector)
            self._free.put(detector)

        self.size = pool_size
        print(f"MediaPipe pool ready: {pool_size} detectors.", flush=True)

    def detect(self, mp_image):
        """Run detection on one image, returning landmarks or None."""
        detector = self._free.get()
        try:
            result = detector.detect(mp_image)
        finally:
            self._free.put(detector)

        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self):
        for detector in self._all:
            try:
                detector.close()
            except Exception:
                pass
        self._all.clear()

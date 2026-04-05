"""
High-performance Hand Tracker
- Self-contained (no external threading or queues)
- Low-latency (frame dropping + latest-frame only)
- Async MediaPipe pipeline
"""

import cv2
import mediapipe as mp
import socket
import threading
import os
from typing import Optional

from mediapipe.tasks.python import vision

_MODEL_PATH = "hand_landmarker.task"


class Tracker:
    def __init__(self, config):
        self.config = config

        # threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # state
        self._running = False
        self._latest_frame: Optional[bytes] = None

        # networking
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return

        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"{_MODEL_PATH} not found")

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self._thread.start()

        self._running = True

    def stop(self):
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_frame(self) -> Optional[bytes]:
        """Return latest JPEG frame (or None)."""
        return self._latest_frame

    # ────────────────────────────────────────────────────────────────
    # Internal loop
    # ────────────────────────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

        # 🔥 reduce internal buffering (huge latency win)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        latest = {"frame": None}

        # ─── Callback (runs in MediaPipe thread) ─────────────────────

        def on_result(result, output_image, ts):
            # --- UDP ---
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]

                data = []
                for lm in lms:
                    data.extend([lm.x, lm.y, lm.z])

                msg = ",".join(f"{v:.5f}" for v in data)

                try:
                    self._sock.sendto(
                        msg.encode(),
                        (self.config.udp.ip, self.config.udp.port)
                    )
                except OSError:
                    pass

            # --- PREVIEW ---
            if self.config.mode.name != "PREVIEW":
                return

            img = output_image.numpy_view().copy()

            if result.hand_landmarks:
                h, w, _ = img.shape
                lms = result.hand_landmarks[0]

                # draw landmarks (lightweight)
                for lm in lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)

            # resize early (smaller = faster encode)
            img = cv2.resize(img, (496, 280))

            latest["frame"] = img

        # ─── MediaPipe setup ─────────────────────────────────────────

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=vision.RunningMode.LIVE_STREAM,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.1,
            num_hands=self.config.num_hands,
            result_callback=on_result,
        )

        # ─── Main loop ───────────────────────────────────────────────

        try:
            with HandLandmarker.create_from_options(options) as landmarker:
                while cap.isOpened() and not self._stop_event.is_set():

                    # 🔥 CRITICAL: drop stale frames
                    for _ in range(2):
                        cap.grab()

                    ok, frame = cap.read()
                    if not ok:
                        continue

                    # convert once
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mp_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=frame_rgb
                    )

                    ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

                    landmarker.detect_async(mp_img, ts)

                    # ─── push latest frame (NO QUEUE) ───────────────

                    if self.config.mode.name == "PREVIEW":
                        self._latest_frame = latest["frame"]

                    # tiny sleep prevents CPU burn
                    self._stop_event.wait(0.001)

        finally:
            cap.release()
            self._sock.close()
            self._running = False
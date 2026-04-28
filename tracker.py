import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pyautogui
import socket

class HandTrackerConfig:
    def __init__(
        self,
        mode: str = "preview",  # "preview" shows cv2 + landmarks, "live" only sends UDP
        udp_ip: str = "127.0.0.1",
        udp_port: int = 5055,
        camera_index: int = 0,
        model_path: str = "hand_landmarker.task",
        min_detection_confidence: float = 0.4,
        min_presence_confidence: float = 0.4,
        min_tracking_confidence: float = 0.2,
        num_hands: int = 1,
    ):
        self.mode = mode
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.camera_index = camera_index
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_hands = num_hands


class HandTracker:
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    def __init__(self, config: HandTrackerConfig):
        self.config = config
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.screen_width, self.screen_height = pyautogui.size()
        self.latest_image = None

        # Initialize camera
        self.cap = cv2.VideoCapture(self.config.camera_index, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.screen_height)

        # Initialize MediaPipe HandLandmarker
        self.options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.config.model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            num_hands=self.config.num_hands,
            result_callback=self._result_callback
        )
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

    def _map_to_screen(self, x, y):
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        screen_x = center_x - (x - 0.5) * self.screen_width * 2
        screen_y = center_y + (y - 0.5) * self.screen_height * 2
        return screen_x, screen_y

    def _result_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result.hand_landmarks:
            img = output_image.numpy_view().copy()
            hand_landmarks = result.hand_landmarks[0]
            if self.config.mode == "preview":
                # Draw landmarks
                for lm in hand_landmarks:
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)

                # Draw connections
                for start, end in self.HAND_CONNECTIONS:
                    x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
                    x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Send UDP
            landmark_data = []
            for lm in hand_landmarks:
                landmark_data.extend([lm.x, lm.y, lm.z])
            msg = ','.join(f'{v:.6f}' for v in landmark_data)

            try:
                self.sock.sendto(msg.encode('utf-8'), (self.config.udp_ip, self.config.udp_port))
            except Exception as e:
                print(f"UDP send error: {e}")

            self.latest_image = img
        else:
            self.latest_image = output_image.numpy_view()

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            self.landmarker.detect_async(mp_image, timestamp_ms)

            if self.config.mode == "preview" and self.latest_image is not None:
                cv2.imshow("Hand Tracking", self.latest_image)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()


if __name__ == "__main__":
    config = HandTrackerConfig(mode="preview")  # "live" or "preview"
    tracker = HandTracker(config)
    tracker.run()
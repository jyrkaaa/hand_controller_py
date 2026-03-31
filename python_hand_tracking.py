import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pyautogui
import socket

# UDP settings
UDP_IP = "127.0.0.1"  # Change to your Unity machine's IP if needed
UDP_PORT = 5055         # Use a free port, match in Unity
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Model path (downloaded from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
threshold = 10

latest_image = None

# Global variables for throttling mouse movement
last_x, last_y = None, None

def map_to_screen(x, y):
    center_x = screen_width / 2
    center_y = screen_height / 2
    screen_x = center_x - (x - 0.5) * screen_width * 2
    screen_y = center_y + (y - 0.5) * screen_height * 2
    return screen_x, screen_y

def move_mouse(screen_x, screen_y):
    pyautogui.moveTo(screen_x, screen_y, duration=0)

# Callback function to handle results and draw overlay
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_image, last_x, last_y
    if result.hand_landmarks:
        # Draw landmarks and connections on the image
        img = output_image.numpy_view().copy()
        hand_landmarks = result.hand_landmarks[0]
        # Draw points
        for lm in hand_landmarks:
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
        # Draw connections (using MediaPipe's hand connections)
        HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),      # Thumb
            (0,5),(5,6),(6,7),(7,8),      # Index
            (5,9),(9,10),(10,11),(11,12), # Middle
            (9,13),(13,14),(14,15),(15,16), # Ring
            (13,17),(17,18),(18,19),(19,20), # Pinky
            (0,17) # Palm base
        ]
        for start, end in HAND_CONNECTIONS:
            x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
            x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- UDP SEND: send all 21 landmarks as x1,y1,x2,y2,...,x21,y21,z1,z2,...z21 ---
        # Format: x1,y1,z1,x2,y2,z2,...,x21,y21,z21 (normalized floats)
        landmark_data = []
        for lm in hand_landmarks:
            landmark_data.extend([lm.x, lm.y, lm.z])
        # Convert to comma-separated string
        msg = ','.join(f'{v:.6f}' for v in landmark_data)
        try:
            sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
        except Exception as e:
            print(f"UDP send error: {e}")

        latest_image = img
    else:
        latest_image = output_image.numpy_view()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.2,
    num_hands=1,
    result_callback=print_result
)
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        #flipped = cv2.flip(image, 1)  # Flip the image horizontally for a mirror view
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # Debug
        if latest_image is not None:
            cv2.imshow('Hand Tracking1', latest_image)  # Flip back for correct orientation
        #else:
        #    cv2.imshow('Hand Tracking2', image)  # Fallback to flipped raw image
        

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
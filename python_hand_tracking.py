# ...existing code...

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pyautogui
import threading

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
        # Use only the middle finger bottom landmark (index 9)
        middle_finger_bottom = result.hand_landmarks[0][9]
        mp_x, mp_y = middle_finger_bottom.x, middle_finger_bottom.y
        screen_x, screen_y = map_to_screen(mp_x, mp_y)
        
        # Throttle and move mouse only if position changed significantly
        if last_x is None or abs(screen_x - last_x) > threshold or abs(screen_y - last_y) > threshold:
            threading.Thread(target=move_mouse, args=(screen_x, screen_y)).start()
            last_x, last_y = screen_x, screen_y
        
    
    latest_image = output_image.numpy_view()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
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
        else:
            cv2.imshow('Hand Tracking2', image)  # Fallback to flipped raw image
        
        
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
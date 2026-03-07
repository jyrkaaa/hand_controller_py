import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Initialize MediaPipe Hand Landmarker
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)
drawing_utils = vision.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)  # Default webcam

# Frame dimensions (adjust for your webcam)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
SENSITIVITY = 0.5  # Scale movement (0-1)

frame_count = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if elapsed_time > 0:  # Avoid division by zero
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
    
    #Optional: Reset every second for real-time display
    if elapsed_time >= 1.0:
        frame_count = 0
        start_time = current_time
    
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(image)

    if results.hand_landmarks:
        # Get the first detected hand
        hand_landmarks = results.hand_landmarks[0]

        # Calculate the center of the hand using average of all landmark positions
        cx = sum([lm.x for lm in hand_landmarks]) / len(hand_landmarks)
        cy = sum([lm.y for lm in hand_landmarks]) / len(hand_landmarks)

        # Convert normalized coordinates to pixel coordinates
        center_x = cx * FRAME_WIDTH
        center_y = cy * FRAME_HEIGHT
        print("Hand center (pixels): ({:.2f}, {:.2f})".format(center_x, center_y))

        # Map to joystick (-1 to 1)
        #joystick_x = ((center_x - CENTER_X) / CENTER_X) * SENSITIVITY
        #joystick_y = ((center_y - CENTER_Y) / CENTER_Y) * SENSITIVITY  # Invert if needed
        #joystick_x = max(-1, min(1, joystick_x))
        #joystick_y = max(-1, min(1, joystick_y))

        # Optional: Draw hand landmarks for debug
        drawing_utils.draw_landmarks(frame, hand_landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS)

    else:
        joystick_x, joystick_y = 0, 0  # No hand detected

    # Display frame (optional)
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
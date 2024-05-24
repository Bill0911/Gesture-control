import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import time

# Get screen size
screen_width, screen_height = pyautogui.size()

# Creating the hand recognizer model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,  # Lower complexity for performance
    min_detection_confidence=0.70,  # Adjusted for balance
    min_tracking_confidence=0.70,
    max_num_hands=1  # Limit the number of hands
)

# Starting webcam to capture with lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower width for performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Lower height for performance

# Smoothing parameters
prev_mapped_x, prev_mapped_y = None, None

def move_mouse_in_steps(start_x, start_y, end_x, end_y, steps=10):
    for step in range(1, steps + 1):
        t = step / steps
        current_x = start_x + (end_x - start_x) * t
        current_y = start_y + (end_y - start_y) * t
        pyautogui.moveTo(current_x, current_y)
        time.sleep(0.001)  # Sleep briefly to allow the movement to be visible

def is_finger_extended(hand_landmarks, finger_tip_id, finger_dip_id):
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_dip_id].y

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def hand_tracking():
    global prev_mapped_x, prev_mapped_y
    target_fps = 60
    duration = 1 / target_fps

    while True:
        start_time = time.time()
        # Read video frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Flip image for mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB image
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processing RGB image
        process = hands.process(frameRGB)

        # Get the height and width of the frame
        h, w, _ = frame.shape

        click_detected = False

        if process.multi_hand_landmarks:
            for hand_no, handlm in enumerate(process.multi_hand_landmarks):
                handedness = process.multi_handedness[hand_no].classification[0].label
                landmarkList = [[_id, int(lm.x * w), int(lm.y * h)] for _id, lm in enumerate(handlm.landmark)]

                if landmarkList:
                    index_tip = landmarkList[8][1:]
                    middle_tip = landmarkList[12][1:]
                    distance_between_fingers = calculate_distance(index_tip, middle_tip)

                    # Move the mouse only if the fingers are fully extended and close to each other
                    index_extended = is_finger_extended(handlm, 8, 6)
                    middle_extended = is_finger_extended(handlm, 12, 10)

                    if distance_between_fingers < 30 and index_extended and middle_extended:
                        avg_x = (index_tip[0] + middle_tip[0]) // 2
                        avg_y = (index_tip[1] + middle_tip[1]) // 2

                        mapped_x = np.interp(avg_x, (0, w), (0, screen_width))
                        mapped_y = np.interp(avg_y, (0, h), (0, screen_height))

                        if prev_mapped_x is not None and prev_mapped_y is not None:
                            # Move the mouse in smaller steps
                            move_mouse_in_steps(prev_mapped_x, prev_mapped_y, mapped_x, mapped_y, steps=3)

                        prev_mapped_x, prev_mapped_y = mapped_x, mapped_y

                    if index_extended and not middle_extended:
                        pyautogui.click(button='left')
                        time.sleep(0.2)  # Add a small delay to prevent multiple clicks
                        click_detected = True
                    elif middle_extended and not index_extended:
                        pyautogui.click(button='right')
                        time.sleep(0.2)  # Add a small delay to prevent multiple clicks
                        click_detected = True

        end_time = time.time()
        elapsed_time = end_time - start_time
        time_to_sleep = max(0, duration - elapsed_time)
        time.sleep(time_to_sleep)

# Create and start the thread
hand_tracking_thread = threading.Thread(target=hand_tracking)
hand_tracking_thread.start()

# Wait for the thread to finish
hand_tracking_thread.join()

cap.release()
cv2.destroyAllWindows()

import cv2
import time
import mediapipe as mp
from math import hypot
import numpy as np
import pyautogui
from tkinter import messagebox

# Creating the hand recognizer model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2,
)

Draw = mp.solutions.drawing_utils

# Starting webcam to capture
cap = cv2.VideoCapture(0)

pyautogui.FAILSAFE = False

def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

last_index_y = None
scroll_cooldown_time = 0.1  # Reduced cooldown for scroll actions
last_scroll_time = time.time()

gesture_cooldown_time = 0.1  # Reduced cooldown for gestures
last_gesture_time = time.time()

def fsm_gesture_control(landmarkList_left, landmarkList_right):
    global last_gesture_time, last_scroll_time, last_index_y

    left_thumb_tip = landmarkList_left[4][1:]
    left_index_tip = landmarkList_left[8][1:]
    left_middle_tip = landmarkList_left[12][1:]
    left_ring_tip = landmarkList_left[16][1:]
    left_pinky_tip = landmarkList_left[20][1:]
    left_wrist = landmarkList_left[0][1:]

    right_thumb_tip = landmarkList_right[4][1:]
    right_index_tip = landmarkList_right[8][1:]
    right_middle_tip = landmarkList_right[12][1:]
    right_ring_tip = landmarkList_right[16][1:]
    right_pinky_tip = landmarkList_right[20][1:]
    right_wrist = landmarkList_right[0][1:]

    current_time = time.time()

    # Right hand gesture logic
    min_scroll_distance = 20  # Minimum distance for scrolling
    if current_time - last_scroll_time > scroll_cooldown_time:
        if right_index_tip[1] < right_middle_tip[1] and (last_index_y is None or abs(right_index_tip[1] - last_index_y) > min_scroll_distance):
            print("Scrolling up")
            pyautogui.scroll(380)
            last_scroll_time = current_time
        elif right_index_tip[1] > right_middle_tip[1] and (last_index_y is None or abs(right_index_tip[1] - last_index_y) > min_scroll_distance):
            print("Scrolling down")
            pyautogui.scroll(-50)
            last_scroll_time = current_time
        elif right_pinky_tip[1] < right_ring_tip[1] and right_pinky_tip[1] > right_index_tip[1]:
            print("Switching tabs")
            pyautogui.keyDown("ctrl")
            pyautogui.press("tab")
            pyautogui.keyUp("ctrl")
            time.sleep(1)
            last_scroll_time = current_time
        last_index_y = right_index_tip[1]

    if current_time - last_gesture_time > gesture_cooldown_time:
        # Gesture for zooming in (hands together like praying)
        distance_between_hands = calculate_distance(left_index_tip, right_index_tip)
        if distance_between_hands < 40:
            pyautogui.hotkey("ctrl", "+")
            last_gesture_time = current_time
        elif distance_between_hands > 100 and calculate_distance(left_wrist, right_wrist) < 20:
            pyautogui.hotkey("ctrl", "-")
            last_gesture_time = current_time

while True:
    # Read video frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing RGB image
    process = hands.process(frameRGB)

    landmarkList_left = []
    landmarkList_right = []

    if process.multi_hand_landmarks:
        for hand_no, handlm in enumerate(process.multi_hand_landmarks):
            # Identify if the hand is left or right
            handedness = process.multi_handedness[hand_no].classification[0].label

            landmarks = []
            for _id, lm in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([_id, cx, cy])

            # Draw landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

            if handedness == "Left":
                landmarkList_left = landmarks

                left_thumb_tip = landmarkList_left[4][1:]
                left_index_tip = landmarkList_left[8][1:]
                left_middle_tip = landmarkList_left[12][1:]

                volumeup_distance = calculate_distance(left_thumb_tip, left_index_tip)
                volumedown_distance = calculate_distance(left_thumb_tip, left_middle_tip)
                # Adjust volume for left hand
                volumeup_level = np.interp(volumeup_distance, [0, 220], [0, 100])
                if volumeup_level < 10:
                    pyautogui.press("volumeup")

                volumedown_level = np.interp(volumedown_distance, [0, 220], [0, 100])
                if volumedown_level < 10:
                    pyautogui.press("volumedown")

            elif handedness == "Right":
                landmarkList_right = landmarks

                right_index_tip = landmarkList_right[8][1:]
                right_middle_tip = landmarkList_right[12][1:]
                right_ring_tip = landmarkList_right[16][1:]
                right_pinky_tip = landmarkList_right[20][1:]

                # Improved right hand gesture logic
                min_scroll_distance = 20  # Minimum distance for scrolling
                if time.time() - last_scroll_time > scroll_cooldown_time:
                    if right_index_tip[1] < right_middle_tip[1] and (last_index_y is None or abs(right_index_tip[1] - last_index_y) > min_scroll_distance):
                        print("Scrolling up")
                        pyautogui.scroll(380)
                        last_scroll_time = time.time()
                    elif right_index_tip[1] > right_middle_tip[1] and (last_index_y is None or abs(right_index_tip[1] - last_index_y) > min_scroll_distance):
                        print("Scrolling down")
                        pyautogui.scroll(-50)
                        last_scroll_time = time.time()
                    elif right_pinky_tip[1] < right_ring_tip[1] and right_pinky_tip[1] > right_index_tip[1]:
                        print("Switching tabs")
                        pyautogui.keyDown("ctrl")
                        pyautogui.press("tab")
                        pyautogui.keyUp("ctrl")
                        time.sleep(1)
                        last_scroll_time = time.time()
                    last_index_y = right_index_tip[1]

        if landmarkList_left and landmarkList_right:
            fsm_gesture_control(landmarkList_left, landmarkList_right)

    # Display the image
    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
        if result:
            break

cap.release()
cv2.destroyAllWindows()

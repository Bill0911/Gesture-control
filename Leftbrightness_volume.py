import cv2
import time
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
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
last_scroll_time = time.time()
scroll_coolDown = 1  # Cooldown period for scroll actions in seconds
min_scroll_distance = 20  # Minimum movement for scroll action

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

    landmarkList = []
    if process.multi_hand_landmarks:
        for hand_no, handlm in enumerate(process.multi_hand_landmarks):
            # Identify if the hand is left or right
            handedness = process.multi_handedness[hand_no].classification[0].label
            
            for _id, lm in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarkList.append([_id, cx, cy])

            # Draw landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

            if landmarkList:
                # Get coordinates for fingers
                thumb_base = landmarkList[2]
                thumb_tip = landmarkList[4]
                index_pip = landmarkList[6]
                index_tip = landmarkList[8]
                middle_pip = landmarkList[10]
                middle_tip = landmarkList[12]
                ring_pip = landmarkList[14]
                ring_tip = landmarkList[16]
                pinky_pip = landmarkList[18]
                pinky_tip = landmarkList[20]


                if last_index_y is None:
                    last_index_y = index_tip[2]

                # Calculate distances
                volumeup_distance = calculate_distance(thumb_tip, index_tip)
                volumedown_distance = calculate_distance(thumb_tip, middle_tip)
                zoom_up_distance = calculate_distance(thumb_tip, ring_tip)
                zoom_down_distance = calculate_distance(thumb_tip, pinky_tip)
                
                # Left hand gestures
                if handedness == "Left":
                    # Adjust volume for left hand
                    volumeup_level = np.interp(volumeup_distance, [0, 220], [0, 100])
                    if volumeup_level < 10:
                        pyautogui.press("volumeup")
                    volumedown_level = np.interp(volumedown_distance, [0, 220], [0, 100])
                    if volumedown_level < 10:
                        pyautogui.press("volumedown")

                    # Adjust zoom for left hand
                    zoom_up_level = np.interp(zoom_up_distance, [0, 220], [0, 100])
                    if zoom_up_level < 10:
                        pyautogui.hotkey("ctrl", "+")
                    zoom_down_level = np.interp(zoom_down_distance, [0, 220], [0, 100])
                    if zoom_down_level < 10:
                        pyautogui.hotkey("ctrl", "-")
                
                # Right hand gestures
                elif handedness == "Right":
                    current_time = time.time()
                    if current_time - last_scroll_time > scroll_coolDown:
                        if (index_tip[2] < middle_tip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance):
                            print("Scrolling up")
                            pyautogui.scroll(380)
                            last_scroll_time = current_time
                        elif (thumb_tip[2] < index_tip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance):
                            print("Scrolling down")
                            pyautogui.scroll(-50)
                            last_scroll_time = current_time
                        elif pinky_tip[2] < pinky_pip[2] and pinky_tip[2] > index_tip[2]:
                            print("Switching tabs")
                            pyautogui.keyDown("ctrl")
                            pyautogui.press("tab")
                            pyautogui.keyUp("ctrl")
                            time.sleep(1)
                            last_switch = time.time()
                        elif (thumb_tip[2] > thumb_base[2] and index_tip[2] < thumb_tip[2]):
                            print("Action for thumb and index gesture")

    # Display the image
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
        if result:
            break

cap.release()
cv2.destroyAllWindows()

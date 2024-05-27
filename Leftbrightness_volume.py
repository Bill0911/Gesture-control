import cv2
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


def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])


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
            if handedness == "Right":
                continue

            for _id, lm in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarkList.append([_id, cx, cy])

            # Draw landmarks
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

            if landmarkList:
                # Get coordinates for fingers
                thumb_tip = landmarkList[4][1:]
                index_tip = landmarkList[8][1:]
                middle_tip = landmarkList[12][1:]
                ring_tip = landmarkList[16][1:]
                pinky_tip = landmarkList[20][1:]

                # Calculate distances
                volumeup_distance = calculate_distance(thumb_tip, index_tip)
                volumedown_distance = calculate_distance(thumb_tip, middle_tip)

                zoom_up_distance = calculate_distance(thumb_tip, ring_tip)
                zoom_down_distance = calculate_distance(thumb_tip, pinky_tip)

                # Adjust volume
                volumeup_level = np.interp(volumeup_distance, [15, 220], [0, 100])
                if volumeup_level < 10:
                    pyautogui.press("volumeup")
                volumedown_level = np.interp(volumedown_distance, [15, 220], [0, 100])
                if volumedown_level < 10:
                    pyautogui.press("volumedown")

                # Adjust volume
                zoom_up_level = np.interp(zoom_up_distance, [15, 220], [0, 100])
                if zoom_up_level < 10:
                    pyautogui.hotkey("ctrl","+")
                zoom_down_level = np.interp(zoom_down_distance, [15, 220], [0, 100])
                if zoom_down_level < 10:
                    pyautogui.hotkey("ctrl","-")

    # Display the image
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
        if result:
            break

cap.release()
cv2.destroyAllWindows()

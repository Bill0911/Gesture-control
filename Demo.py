import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
import pyautogui
import time
import tkinter as tk
from tkinter import messagebox

# Create the hand recognizer model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

Draw = mp.solutions.drawing_utils

# Start the webcam to capture video
cap = cv2.VideoCapture(0)

# Initialize variables for the scroll up-down feature
last_scroll_time = 0
scroll_coolDown = 1.5
min_scroll_distance = 0.05
last_index_y = None
last_tab_time = 0
tab_coolDown = 1.5
new_tab_opened = False

# Initialize variables for the brightness adjustment feature
last_brightness_time = None

last_switch = time.time()

def confirm_exit():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    result = messagebox.askquestion("Exit Confirmation", "You still have a chance to be moved on. Are you sure to stop using our lovely software?", icon='warning')
    root.destroy()  # Destroy the main window
    return result == 'yes'

while True:
    # Read the video frame
    _, frame = cap.read()

    # Flip the image
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    Process = hands.process(frameRGB)

    landmarkList = []
    
    # If hands are detected in the image
    if Process.multi_hand_landmarks:
        # Recognize hand landmarks
        for handlm in Process.multi_hand_landmarks:
            # Get the handedness of the detected hand
            handedness = Process.multi_handedness[0].classification[0].label
            # Only process the landmarks if the detected hand is the right hand
            if handedness == 'Right':
                for _id, landmarks in enumerate(handlm.landmark):
                    # Store the height and width of the image
                    height, width, color_channels = frame.shape

                    # Calculate and append the x, y coordinates of the hand landmarks to the list
                    x, y = int(landmarks.x * width), int(landmarks.y * height)
                    landmarkList.append([_id, x, y])

                # Draw the hand landmarks
                Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    # If the landmarks list is not empty
    if landmarkList != []:
        # Scroll up or down
        if len(landmarkList) >= 21:  # Make sure the index finger tip and middle finger tip are detected
            index_tip = landmarkList[8]
            thumb_tip = landmarkList[4]
            thumb_base = landmarkList[2]
            index_pip = landmarkList[6]
            pinky_tip = landmarkList[20]
            pinky_pip = landmarkList[18]
            ring_tip = landmarkList[16]
            ring_pip = landmarkList[14]
            middle_tip = landmarkList[12]
            middle_pip = landmarkList[10]

            if last_index_y is None:
                last_index_y = index_tip[2]

            if time.time() - last_scroll_time > scroll_coolDown:
                if index_tip[2] < middle_tip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance:
                    print("Scrolling up")
                    pyautogui.scroll(380)
                    last_scroll_time = time.time()
            elif thumb_tip[2] < index_tip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance:
                print("Scrolling down")
                pyautogui.scroll(-50)
                last_scroll_time = time.time()
            elif pinky_tip[2] < pinky_pip[2] and pinky_tip[2] > index_tip[2]:
                print("Switching tabs")
                pyautogui.keyDown('ctrl')
                pyautogui.press('tab')
                pyautogui.keyUp('ctrl')
                time.sleep(1)
                last_switch = time.time()
            elif thumb_tip[2] > thumb_base[2] and index_tip[2] < thumb_tip[2] and index_tip[2] < index_pip[2]:
               if confirm_exit(): 
                  print("Turning off webcam")
                  break
               else:
                  print("Continuing webcam")
            

        last_index_y = index_tip[2]

    # If the user wants to turn off the webcam, press 'q'
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
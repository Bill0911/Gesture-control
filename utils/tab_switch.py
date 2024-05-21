from time import time

import pyautogui
from config import Config

last_switch = time()


def process(cv, frame, process):
    config = Config()

    global last_switch

    if process.multi_hand_landmarks:
        for hand_landmarks in process.multi_hand_landmarks:

            if config.DEBUG:
                # Draw hand landmarks
                cv.draw.draw_landmarks(
                    frame, hand_landmarks, cv.mp_hands.HAND_CONNECTIONS
                )

            # Get the landmarks for the index finger tip and PIP
            index_tip = hand_landmarks.landmark[
                cv.mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]
            index_pip = hand_landmarks.landmark[
                cv.mp_hands.HandLandmark.INDEX_FINGER_PIP
            ]

            # Check if the index finger is raised and at least 0.5 seconds have passed since the last tab switch
            if index_pip.y - index_tip.y > 0.1 and time() - last_switch > 0.5:
                if config.DEBUG:
                    print("Switching tabs")  # Debugging line
                # Press down 'ctrl'
                pyautogui.keyDown("ctrl")
                # Press 'tab'
                pyautogui.press("tab")
                # Release 'ctrl'
                pyautogui.keyUp("ctrl")
                # Update the time of the last tab switch
                last_switch = time()
            else:  # Test if the switching tabs feature is stable or not
                if config.DEBUG:
                    print("Not switching tabs")  # Debugging line
                    print(
                        f"index_tip.y: {index_tip.y}, index_pip.y: {index_pip.y}, time since last switch: {time() - last_switch}"
                    )

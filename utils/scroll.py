import pyautogui
from config import Config


def process(cv, frame, process):
    config = Config()

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
            middle_tip = hand_landmarks.landmark[
                cv.mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            ]
            pyautogui.FAILSAFE = False

            # Check if the index finger is above the middle finger
            if index_tip.y < middle_tip.y:
                if config.DEBUG:
                    print("Scrolling up")  # Debugging line
                pyautogui.scroll(50)  # Scroll up

                # Check if the middle finger is above the index finger
            elif middle_tip.y < index_tip.y:
                if config.DEBUG:
                    print("Scrolling down")  # Debugging line
                pyautogui.scroll(-50)  # Scroll down

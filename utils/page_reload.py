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

            thumb_tip = hand_landmarks.landmark[cv.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[
                cv.mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            if abs(thumb_tip.y - index_tip.y) < 0.05:
                if config.DEBUG:
                    print("reload page")  # Debugging line

                # Press down 'ctrl'
                pyautogui.keyDown("ctrl")
                # Press 'tab'
                pyautogui.press("r")
                # Release 'ctrl'
                pyautogui.keyUp("ctrl")

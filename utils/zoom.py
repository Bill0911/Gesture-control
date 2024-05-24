import pyautogui
from config import Config


def process(cv, frame, process):
    config = Config()

    if process.multi_hand_landmarks:
        for hand, hand_type in zip(
            process.multi_hand_landmarks, process.multi_handedness
        ):
            if hand_type.classification[0].label != "Left":
                continue

            if config.DEBUG:
                # Draw hand landmarks
                cv.draw.draw_landmarks(frame, hand, cv.mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand.landmark[cv.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand.landmark[cv.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # if abs(thumb_tip.y - index_tip.y) < 0.05:
            #     if config.DEBUG:
            #         print("zoom in")
            #
            #     # Press down 'win'
            #     pyautogui.keyDown("win")
            #     # Press '+'
            #     pyautogui.press("+")
            #     # Release 'win'
            #     pyautogui.keyUp("win")
            #
            # elif thumb_tip.y < index_tip.y:
            #     if config.DEBUG:
            #         print("zoom out")
            #
            #     # Press down 'win'
            #     pyautogui.keyDown("win")
            #     # Press '+'
            #     pyautogui.press("+")
            #     # Release 'win'
            #     pyautogui.keyUp("win")

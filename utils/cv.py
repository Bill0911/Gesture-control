import cv2
import mediapipe as mp

from config import Config


class CV:
    def __init__(self) -> None:
        self.cam = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=2,
        )
        self.draw = mp.solutions.drawing_utils

    def __call__(self, *porcesses):
        config = Config()

        while True:
            _, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_process = self.hands.process(frameRGB)

            for process in porcesses:
                if callable(process):
                    process(self, frame, hand_process)

            if config.DEBUG:
                cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cam.release()
        cv2.destroyAllWindows()

        self.hands.close()

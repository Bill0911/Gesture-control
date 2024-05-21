import cv2
from math import hypot
import numpy as np
import screen_brightness_control as sbc

from config import Config


def process(cv, frame, process):
    config = Config()

    landmarks = []

    build_landmarks(cv, frame, process, landmarks)

    if landmarks != []:
        # store x,y coordinates of tip of thumb
        x_1, y_1 = landmarks[4][1], landmarks[4][2]

        # store x,y coordinates of tip of index finger
        x_2, y_2 = landmarks[8][1], landmarks[8][2]

        if config.DEBUG:
            # draw circle on thumb and index finger tip
            cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)

            # draw line from tip of thumb to tip of index finger
            cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

        # calculate square root of the sum of
        # squares of the specified arguments.
        L = hypot(x_2 - x_1, y_2 - y_1)

        # 1-D linear interpolant to a function
        # with given discrete data points
        # (Hand range 15 - 220, Brightness
        # range 0 - 100), evaluated at length.
        b_level = np.interp(L, [15, 220], [0, 100])

        # set up brightness
        sbc.set_brightness(int(b_level))


def build_landmarks(cv, frame, process, landmarks):
    config = Config()

    if process.multi_hand_landmarks:
        for hand_mark in process.multi_hand_landmarks:
            for i, landmark in enumerate(hand_mark.landmark):
                height, width, color_channels = frame.shape

                x, y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append([i, x, y])

            if config.DEBUG:
                cv.draw.draw_landmarks(frame, hand_mark, cv.mp_hands.HAND_CONNECTIONS)

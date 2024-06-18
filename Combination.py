import cv2
import threading
import pyautogui
import mediapipe as mp
import numpy as np

from math import hypot, sqrt
from time import time
from tkinter import messagebox
from pykalman import KalmanFilter
from pynput.keyboard import Key, Controller
from mediapipe.python.solutions.hands import HandLandmark

# Constants and Initialization
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2,
)

draw = mp.solutions.drawing_utils

ROI_TOP = 0.1
ROI_BOTTOM = 0.9
ROI_LEFT = 0.1
ROI_RIGHT = 0.9

SMOTH_FACTOR = 0.8
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SCROLL_AMOUNT = 40
SCROLL_ITERATIONS = 10
prev_x, prev_y = 0.0, 0.0
last_scroll_time = time()
scroll_cool_down = 1
min_scroll_distance = 20
mouse_control_active = False
mouse_is_down = False
exit_program = False
is_exit_thread = False
mode = "MAIN"

initial_state = np.array([0, 0])
initial_state_uncertainty = np.eye(2)

kf = KalmanFilter(
    transition_matrices=np.eye(2),
    observation_matrices=np.eye(2),
    initial_state_mean=initial_state,
    initial_state_covariance=initial_state_uncertainty,
)


# Gesture Detection Functions
def calculate_distance(point1, point2):
    return hypot(point2.x - point1.x, point2.y - point1.y)


def detect_two_fingers_up(hand_landmarks):
    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP]

    return index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y


def detect_thumb_near_index_mcp(hand_landmarks, height, width):
    index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP]
    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]

    index_mcp_x = int(index_mcp.x * width)
    index_mcp_y = int(index_mcp.y * height)
    thumb_x = int(thumb_tip.x * width)
    thumb_y = int(thumb_tip.y * height)

    distance = np.sqrt((index_mcp_x - thumb_x) ** 2 + (index_mcp_y - thumb_y) ** 2)
    return distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y


def confirm_exit():
    global exit_program, is_exit_thread

    is_exit_thread = True

    result = messagebox.askyesno("Confirm Exit", "Do you actually want to turn it off?")
    if result:
        exit_program = True

    is_exit_thread = False

    return


# Main Application Loop
def main():
    global prev_x, prev_y, last_scroll_time, mouse_control_active, mouse_is_down, is_exit_thread, mode

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if exit_program:
            break

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        process = hands.process(frameRGB)

        height, width, _ = frame.shape

        top_left = (int(width * ROI_LEFT), int(height * ROI_TOP))
        bottom_right = (int(width * ROI_RIGHT), int(height * ROI_BOTTOM))

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow("Image", frameRGB)

        if process.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                process.multi_hand_landmarks, process.multi_handedness
            ):
                match mode:
                    case "MAIN":
                        handle_main_mode(
                            hand_landmarks, handedness, frame, height, width
                        )
                    case "GAMING":
                        handle_gaming_mode(hand_landmarks, handedness, frame)

    cap.release()
    cv2.destroyAllWindows()


def handle_main_mode(hand_landmarks, handedness, frame, height, width):
    label = handedness.classification[0].label
    draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if label == "Left":
        handle_main_left_hand_gestures(hand_landmarks)

    if label == "Right":
        handle_main_right_hand_gestures(hand_landmarks, height, width, frame)


def handle_gaming_mode(hand_landmarks, handedness, frame):
    label = handedness.classification[0].label
    draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if label == "Left":
        handle_gaming_left_hand_gestures(hand_landmarks)

    if label == "Right":
        handle_gaming_right_hand_gestures(hand_landmarks)


def handle_main_left_hand_gestures(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[HandLandmark.PINKY_TIP]

    volumeup_distance = calculate_distance(thumb_tip, index_tip)
    volumedown_distance = calculate_distance(thumb_tip, middle_tip)
    zoom_up_distance = calculate_distance(thumb_tip, ring_tip)
    zoom_down_distance = calculate_distance(thumb_tip, pinky_tip)

    if volumeup_distance < 0.05:
        pyautogui.press("volumeup")
    if volumedown_distance < 0.05:
        pyautogui.press("volumedown")
    if zoom_up_distance < 0.05:
        pyautogui.hotkey("ctrl", "+")
    if zoom_down_distance < 0.05:
        pyautogui.hotkey("ctrl", "-")


def handle_main_right_hand_gestures(hand_landmarks, height, width, frame):
    global last_scroll_time, mouse_control_active, mouse_is_down, mode

    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
    thumb_cmc = hand_landmarks.landmark[HandLandmark.THUMB_CMC]
    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[HandLandmark.PINKY_PIP]
    pinky_mcp = hand_landmarks.landmark[HandLandmark.PINKY_MCP]
    wrist = hand_landmarks.landmark[HandLandmark.WRIST]

    switch_mode_distance = calculate_distance(thumb_tip, thumb_cmc)

    locked_index_finger_distance = calculate_distance(index_tip, wrist)
    locked_middle_finger_distance = calculate_distance(middle_tip, wrist)

    current_time = time()

    thumb_near_index = abs(thumb_tip.y - index_tip.y) < 0.05

    if detect_two_fingers_up(hand_landmarks):
        activate_mouse_control(
            hand_landmarks, middle_tip, thumb_near_index, frame, height, width
        )

    if (
        switch_mode_distance > 0.20
        and thumb_tip.y < thumb_cmc.y
        and locked_index_finger_distance < 0.35
        and locked_middle_finger_distance < 0.30
    ):
        mode = "GAMING"
        print(mode)

    if mouse_control_active:
        turnoff_distance = calculate_distance(thumb_tip, pinky_tip)
        page_refresh_distance = calculate_distance(thumb_tip, middle_tip)

        if current_time - last_scroll_time > scroll_cool_down:
            if (
                pinky_tip.y < pinky_pip.y
                and pinky_tip.y < pinky_mcp.y
                and ring_tip.y > pinky_tip.y
                and abs(pinky_tip.y - index_tip.y) < 0.05
            ):
                pyautogui.hotkey("ctrl", "tab")
                pyautogui.sleep(1)
            elif (
                index_pip.y > index_tip.y
                and middle_tip.y > middle_mcp.y
                and (index_pip.y - index_tip.y) * 10 > 1.05
            ):
                for _ in range(SCROLL_ITERATIONS):
                    pyautogui.scroll(SCROLL_AMOUNT)
                last_scroll_time = current_time
            elif index_pip.y < index_tip.y and (index_pip.y - index_tip.y) * 10 < -1.25:
                for _ in range(SCROLL_ITERATIONS):
                    pyautogui.scroll(-SCROLL_AMOUNT)
                last_scroll_time = current_time
            elif page_refresh_distance < 0.05 and mouse_control_active:
                pyautogui.hotkey("ctrl", "r")
                last_scroll_time = current_time

            if turnoff_distance < 0.05 and not is_exit_thread:
                exit_thread = threading.Thread(target=confirm_exit)
                exit_thread.start()


def activate_mouse_control(
    hand_landmarks, middle_tip, thumb_near_index, frame, height, width
):
    global mouse_control_active, mouse_is_down, kf

    mouse_control_active = True

    norm_x = (middle_tip.x - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)
    norm_y = (middle_tip.y - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)

    if isinstance(kf.initial_state_mean, list):
        kf.initial_state_mean = np.array(kf.initial_state_mean)

    observation = np.array([norm_x, norm_y])
    filtered_state_means, filtered_state_covariances = kf.filter_update(
        kf.initial_state_mean, kf.initial_state_covariance, observation
    )
    filtered_x, filtered_y = filtered_state_means

    screen_x = filtered_x * SCREEN_WIDTH
    screen_y = filtered_y * SCREEN_HEIGHT

    norm_x = min(max(float(filtered_x), 0), 1)
    norm_y = min(max(float(filtered_y), 0), 1)

    screen_width, screen_height = pyautogui.size()
    screen_x = int(screen_width * norm_x)
    screen_y = int(screen_height * norm_y)

    pyautogui.moveTo(screen_x, screen_y)

    if thumb_near_index and not mouse_is_down:
        pyautogui.mouseDown()
        mouse_is_down = True
    elif not thumb_near_index and mouse_is_down:
        pyautogui.mouseUp()
        mouse_is_down = False

    distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y = detect_thumb_near_index_mcp(
        hand_landmarks, height, width
    )
    cv2.circle(frame, (index_mcp_x, index_mcp_y), 10, (0, 255, 255), -1)
    cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)

    if distance < 25:
        pyautogui.mouseDown()
        pyautogui.sleep(0.2)
        pyautogui.mouseUp()

    kf.initial_state_mean = filtered_state_means
    kf.initial_state_covariance = filtered_state_covariances


def handle_gaming_left_hand_gestures(hand_landmarks): ...


def handle_gaming_right_hand_gestures(hand_landmarks):
    global mode

    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
    thumb_cmc = hand_landmarks.landmark[HandLandmark.THUMB_CMC]
    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[HandLandmark.WRIST]

    switch_mode_distance = calculate_distance(thumb_tip, thumb_cmc)

    locked_index_finger_distance = calculate_distance(index_tip, wrist)
    locked_middle_finger_distance = calculate_distance(middle_tip, wrist)

    average_tip = {
        "x": np.mean([thumb_tip.x, thumb_tip.x, ring_tip.x, pinky_tip.x]),
        "y": np.mean([thumb_tip.y, thumb_tip.y, ring_tip.y, pinky_tip.y]),
    }

    if (
        switch_mode_distance > 0.20
        and thumb_tip.y > thumb_cmc.y
        and locked_index_finger_distance < 0.35
        and locked_middle_finger_distance < 0.30
    ):
        mode = "MAIN"
        print(mode)

    keyboard = Controller()

    dx = wrist.x - average_tip["x"]
    dy = wrist.y - average_tip["y"]

    x1, y1, x2, y2 = (0, 0, dx, dy)

    distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if distance > 0.2:
        if abs(x2 - x1) > abs(y2 - y1):
            if x2 > x1:
                keyboard.press(Key.left)
                keyboard.release(Key.left)
                print("left")
            else:
                keyboard.press(Key.right)
                keyboard.release(Key.right)
                print("right")
        else:
            if y2 > y1:
                keyboard.press(Key.up)
                keyboard.release(Key.up)
                print("up")
            else:
                keyboard.press(Key.down)
                keyboard.release(Key.down)
                print("down")


if __name__ == "__main__":
    main_thread = threading.Thread(target=main)
    main_thread.start()
    main_thread.join()

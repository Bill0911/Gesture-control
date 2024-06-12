from time import time
from math import hypot
import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import HandLandmark
import numpy as np
import pyautogui
from tkinter import messagebox
import tkinter as tk
import tkinter.messagebox as messagebox
import threading
from pykalman import KalmanFilter #You guys may need to install this packet



pyautogui.FAILSAFE = False

# Creating the hand recognizer model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2,
)

draw = mp.solutions.drawing_utils

# Define region of interest (ROI) within the webcam frame
ROI_TOP = 0.2  # Top 20% of the frame height
ROI_BOTTOM = 0.8  # Bottom 80% of the frame height
ROI_LEFT = 0.2  # Left 20% of the frame width
ROI_RIGHT = 0.8  # Right 80% of the frame width

# Smoothing parameters
SMOTH_FACTOR = 0.8

# Get screen size
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

SCROLL_AMOUNT = 30
SCROLL_ITERATIONS = 10

prev_x, prev_y = 0.0, 0.0
last_scroll_time = time()
scroll_cool_down = 1
min_scroll_distance = 20
mouse_control_active = False
mouse_is_down = False
exit_program = False
is_exit_thread = False

initial_state = [0, 0]

#define the initial state uncertainty
initial_state_uncertainty = [[1, 0], [0, 1]]

#Initialize the Kalman filter
kf = KalmanFilter(initial_state_mean=initial_state, initial_state_covariance=initial_state_uncertainty)


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
    point1 (object): The first point.
    point2 (object): The second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return hypot(point2.x - point1.x, point2.y - point1.y)


def detect_two_fingers_up(hand_landmarks):
    """
    Detects if the index and middle fingers are raised.

    This function checks if the y-coordinate of the tip of the index and middle fingers is less than the y-coordinate of the MCP joint of the same fingers.
    If so, it means that the fingers are raised (assuming the hand is approximately vertical).

    Args:
    hand_landmarks (object): The hand landmarks object obtained from MediaPipe Hands.

    Returns:
    bool: True if the index and middle fingers are raised, False otherwise.
    """
    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP]

    return index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y


def detect_thumb_near_index_mcp(hand_landmarks, height, width):
    """
    Detects if the thumb is near the MCP joint of the index finger.

    This function calculates the pixel coordinates of the thumb tip and the MCP joint of the index finger.
    It then calculates the Euclidean distance between these two points.

    Args:
    hand_landmarks (object): The hand landmarks object obtained from MediaPipe Hands.
    height (int): The height of the frame.
    width (int): The width of the frame.

    Returns:
    tuple: A tuple containing the distance between the thumb tip and the MCP joint of the index finger,
           the x and y coordinates of the MCP joint of the index finger, and the x and y coordinates of the thumb tip.
    """
    index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP]
    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]

    index_mcp_x = int(index_mcp.x * width)
    index_mcp_y = int(index_mcp.y * height)
    thumb_x = int(thumb_tip.x * width)
    thumb_y = int(thumb_tip.y * height)

    distance = np.sqrt((index_mcp_x - thumb_x) ** 2 + (index_mcp_y - thumb_y) ** 2)
    return distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y


def confirm_exit():
    """
    Confirm the exit of the program.

    This function prompts the user with a message box asking if they want to exit the program.
    If the user confirms, it sets the global variable 'exit_program' to True, signaling the program to terminate.

    Returns:
    None
    """
    global exit_program, is_exit_thread

    is_exit_thread = True

    result = messagebox.askyesno("Confirm Exit", "Do you actually want turn it off ?")
    if result:
        exit_program = True

    is_exit_thread = False

    return


class KalmanFilter:
    """
    Implements a one-dimensional Kalman filter.

    A Kalman filter is an algorithm that uses a series of measurements observed over time,
    containing statistical noise and other inaccuracies, and produces estimates of unknown variables
    that tend to be more accurate than those based on a single measurement alone.

    Attributes:
    process_variance (float): The variance in the process model.
    estimated_measurement_variance (float): The estimated variance of the measurements.
    posteri_estimate (float): The posteriori state estimate.
    posteri_error_estimate (float): The posteriori estimate of the error covariance.
    """

    def __init__(self, process_variance, estimated_measurement_variance):
        """
        Initializes the KalmanFilter class with process variance and estimated measurement variance.

        Args:
        process_variance (float): The variance in the process model.
        estimated_measurement_variance (float): The estimated variance of the measurements.
        """
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def get_estimated_measurement(self, measurement):
        """
        Updates and returns the Kalman Filter's state estimate from the given measurement.

        Args:
        measurement (float): The current measurement.

        Returns:
        float: The updated state estimate.
        """
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (
            priori_error_estimate + self.estimated_measurement_variance
        )
        self.posteri_estimate = priori_estimate + blending_factor * (
            measurement - priori_estimate
        )
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate


kf_x = KalmanFilter(process_variance=1e-5, estimated_measurement_variance=0.3)
kf_y = KalmanFilter(process_variance=1e-5, estimated_measurement_variance=0.3)

class Smoother: #Here I added this to make the cursor way smoother
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def add(self, value):
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)

    def get_average(self):
        return sum(self.data) / len(self.data)

# Initialize the smoother
smoother = Smoother(5)  # Use the last 5 positions for smoothing

def main():
    """
    The main function of the program.

    This function captures video from the webcam, processes each frame, and performs actions based on the detected hand gestures.
    The actions include moving the mouse cursor, clicking, scrolling, switching tabs, and refreshing the page.
    The function also handles the exit of the program when the user confirms it through a message box.

    Returns:
    None
    """
    global prev_x, prev_y, last_scroll_time, mouse_control_active, mouse_is_down, is_exit_thread

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

        if process.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                process.multi_hand_landmarks, process.multi_handedness
            ):
                label = handedness.classification[0].label

                index_mcp = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_MCP
                ]

                index_mcp_x = index_mcp.x
                index_mcp_y = index_mcp.y

                if frame.shape[1] and frame.shape[0]:
                    if index_mcp_x is not None and index_mcp_y is not None:
                        try:
                            #Update the Kalman Filter with the current measurement
                            current_state_estimate, current_state_covariance = kf.filter_update(
                                filtered_state_means[-1], filter_state_covariances[-1], [index_mcp_x, index_mcp_y]
                            )
                            #Get the estimate from the Kalman filter
                            (filtered_state_means, filtered_state_covariances) = kf.filter([index_mcp_x, index_mcp_y])

                            #Add the estimate to the smoother
                            smoother.add(current_state_estimate)

                            #Get the smoothed estimate
                        except:
                            pass

                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if hand_landmarks:
                    thumb_base = hand_landmarks.landmark[HandLandmark.THUMB_MCP]
                    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
                    index_pip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_PIP]
                    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
                    middle_mcp = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP]
                    middle_pip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_PIP]
                    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
                    ring_pip = hand_landmarks.landmark[HandLandmark.RING_FINGER_PIP]
                    ring_tip = hand_landmarks.landmark[HandLandmark.RING_FINGER_TIP]
                    pinky_pip = hand_landmarks.landmark[HandLandmark.PINKY_PIP]
                    pinky_tip = hand_landmarks.landmark[HandLandmark.PINKY_TIP]
                    pinky_mcp = hand_landmarks.landmark[HandLandmark.PINKY_MCP]

                    # left hand
                    volumeup_distance = calculate_distance(thumb_tip, index_tip)
                    volumedown_distance = calculate_distance(thumb_tip, middle_tip)
                    zoom_up_distance = calculate_distance(thumb_tip, ring_tip)
                    zoom_down_distance = calculate_distance(thumb_tip, pinky_tip)

                    # right hand
                    turnoff_distance = calculate_distance(thumb_tip, pinky_tip)
                    page_refresh_distance = calculate_distance(thumb_tip, ring_tip)

                    if label == "Left":
                        if volumeup_distance < 0.05:
                            pyautogui.press("volumeup")
                        if volumedown_distance < 0.05:
                            pyautogui.press("volumedown")
                        if zoom_up_distance < 0.05:
                            pyautogui.hotkey("ctrl", "+")
                        if zoom_down_distance < 0.05:
                            pyautogui.hotkey("ctrl", "-")

                    prev_x, prev_y = 0, 0
                    prev_time, prev_prev_time = 0, 0

                    if label == "Right":
                        current_time = time()
                        thumb_near_index = abs(thumb_tip.y - index_tip.y) < 0.05

                        if detect_two_fingers_up(hand_landmarks):
                            mouse_control_active = True

                            norm_x = (middle_tip.x - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)
                            norm_y = (middle_tip.y - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)

                            filtered_x = kf_x.get_estimated_measurement(norm_x)
                            filtered_y = kf_y.get_estimated_measurement(norm_y)

                            screen_x = filtered_x * SCREEN_WIDTH
                            screen_y = filtered_y * SCREEN_HEIGHT

                            norm_x = min(max(norm_x, 0), 1)
                            norm_y = min(max(norm_y, 0), 1)

                            screen_width, screen_height = pyautogui.size()
                            screen_x = int(screen_width * norm_x)
                            screen_y = int(screen_height * norm_y)

                            if prev_time and prev_prev_time:
                                velocity_x = (screen_x - prev_x) / (
                                    current_time - prev_time
                                )
                                velocity_y = (screen_y - prev_y) / (
                                    current_time - prev_time
                                )

                                future_x = screen_x + velocity_x * 0.2
                                future_y = screen_y + velocity_y * 0.2

                                pyautogui.moveTo(future_x, future_y)

                            prev_x, prev_y = screen_x, screen_y
                            prev_prev_time, prev_time = prev_time, int(current_time)

                            pyautogui.moveTo(screen_x, screen_y)

                            if thumb_near_index and not mouse_is_down:
                                pyautogui.mouseDown()
                                mouse_is_down = True
                            elif not thumb_near_index and mouse_is_down:
                                pyautogui.mouseUp()
                                mouse_is_down = False

                            distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y = (
                                detect_thumb_near_index_mcp(
                                    hand_landmarks, height, width
                                )
                            )
                            cv2.circle(
                                frame, (index_mcp_x, index_mcp_y), 10, (0, 255, 255), -1
                            )
                            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)

                            if distance < 25:
                                pyautogui.mouseDown()
                                pyautogui.sleep(0.2)
                                pyautogui.mouseUp()

                        elif mouse_control_active:
                            mouse_control_active = False
                            if mouse_is_down:
                                pyautogui.mouseUp()
                                mouse_is_down = False

                        if current_time - last_scroll_time > scroll_cool_down:
                            if (
                                pinky_tip.y < pinky_pip.y
                                and pinky_tip.y < pinky_mcp.y
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

                            elif (
                                index_pip.y < index_tip.y
                                and (index_pip.y - index_tip.y) * 10 < -1.25
                            ):
                                for _ in range(SCROLL_ITERATIONS):
                                    pyautogui.scroll(-SCROLL_AMOUNT)
                                last_scroll_time = current_time

                            elif (
                                page_refresh_distance < 0.05
                                and not mouse_control_active
                            ):
                                pyautogui.hotkey("ctrl", "r")
                                last_scroll_time = current_time

                            if (
                                turnoff_distance < 0.05
                                or cv2.waitKey(1) & 0xFF == ord("q")
                            ) and not is_exit_thread:
                                exit_thread = threading.Thread(target=confirm_exit)
                                exit_thread.start()

        cv2.imshow("Image", frame)

    cap.release()
    cv2.destroyAllWindows()


main_thread = threading.Thread(target=main)
main_thread.start()
main_thread.join()

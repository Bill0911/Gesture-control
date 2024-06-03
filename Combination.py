from time import time, sleep
from math import hypot
import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import HandLandmark
import numpy as np
import pyautogui
from tkinter import messagebox
import threading

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


SCROLL_AMOUNT = 20 #Amount to scroll in each iteration
SCROLL_ITERATIONS = 10 # Number of iterations


SCROLL_AMOUNT = 20  # Amount to scroll in each iteration
SCROLL_ITERATIONS = 10  # Number of iterations





prev_x, prev_y = 0.0, 0.0
last_scroll_time = time()
scroll_cool_down = 1
min_scroll_distance = 20
mouse_control_active = False
mouse_is_down = False

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

class KalmanFilter:
    def __init__(self, process_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def get_estimated_measurement(self, measurement):
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

# Flag to stop the capture thread
stop_thread = False

def capture_and_process():
    global prev_x, prev_y, last_scroll_time, mouse_control_active, mouse_is_down, stop_thread

    cap = cv2.VideoCapture(0)

    while not stop_thread:
        success, frame = cap.read()


    process = hands.process(frameRGB)

    height, width, _ = frame.shape

    top_left = (int(width * ROI_LEFT), int(height * ROI_TOP))
    bottom_right = (int(width * ROI_RIGHT), int(height * ROI_BOTTOM))

    # Draw ROI rectangle
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    if process.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            process.multi_hand_landmarks, process.multi_handedness
        ):
            label = handedness.classification[0].label

            # Get the index finger MCP joint
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Get the x and y coordinates of the index finger MCP joint
            index_mcp_x = index_mcp.x
            index_mcp_y = index_mcp.y

            if frame.shape[1] and frame.shape[0]:
                if index_mcp_x is not None and index_mcp_y is not None:
                    try:
                        index_mcp_x_px = int(index_mcp_x * frame.shape[1])
                        index_mcp_y_px = int(index_mcp_y * frame.shape[0])

                        cv2.circle(
                            frame,
                            (index_mcp_x_px, index_mcp_y_px),
                            10,
                            (0, 255, 255),
                            -1,
                        )
                    except cv2.error as e:
                        print(f"OpenCV error: {e}")
                else:
                    raise ValueError(f"Invalid center for circle")

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

                # Calculate distances
                volumeup_distance = calculate_distance(thumb_tip, index_tip)
                volumedown_distance = calculate_distance(thumb_tip, middle_tip)
                zoom_up_distance = calculate_distance(thumb_tip, ring_tip)
                zoom_down_distance = calculate_distance(thumb_tip, pinky_tip)

                # Left hand gestures
                if label == "Left":
                    # Adjust volume for left hand
                    if volumeup_distance < 0.05:
                        pyautogui.press("volumeup")
                    if volumedown_distance < 0.05:
                        pyautogui.press("volumedown")

                    # Adjust zoom for left hand
                    if zoom_up_distance < 0.05:
                        pyautogui.hotkey("ctrl", "+")
                    if zoom_down_distance < 0.05:
                        pyautogui.hotkey("ctrl", "-")
                # Initialize previous positions and times
                prev_x, prev_y = 0, 0
                prev_prev_x, prev_prev_y = 0, 0
                prev_time, prev_prev_time = 0, 0

                # Right hand gestures
                if label == "Right":
                    current_time = time()
                    thumb_near_index = abs(thumb_tip.y - index_tip.y) < 0.05

                    if detect_two_fingers_up(hand_landmarks):
                        print("mouse control active")
                        mouse_control_active = True
                        # Normalize the coordinates within the frame
                        norm_x = (middle_tip.x - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)
                        norm_y = (middle_tip.y - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)

                        # Filter the positions
                        filtered_x = kf_x.get_estimated_measurement(norm_x)
                        filtered_y = kf_y.get_estimated_measurement(norm_y)

                        # Use the filtered positions for mouse control
                        screen_x = filtered_x * SCREEN_WIDTH
                        screen_y = filtered_y * SCREEN_HEIGHT

                        # Clamp normalized coordinates to [0, 1] range
                        norm_x = min(max(norm_x, 0), 1)
                        norm_y = min(max(norm_y, 0), 1)

                        # Convert normalized coordinates to screen coordinates
                        screen_width, screen_height = pyautogui.size()
                        screen_x = int(screen_width * norm_x)
                        screen_y = int(screen_height * norm_y)

                        # Calculate velocities
                        if prev_time and prev_prev_time:
                            velocity_x = (screen_x - prev_x) / (
                                current_time - prev_time
                            )
                            velocity_y = (screen_y - prev_y) / (
                                current_time - prev_time
                            )

                            # predict future positions
                            future_x = screen_x + velocity_x * 0.2
                            future_y = screen_y + velocity_y * 0.2

                            # move the cursor to the predicted position
                            pyautogui.moveTo(future_x, future_y)

                        # Update previous positions and times
                        prev_prev_x, prev_prev_y = prev_x, prev_y
                        prev_x, prev_y = screen_x, screen_y
                        prev_prev_time, prev_time = prev_time, int(current_time)

                        # Move the mouse
                        pyautogui.moveTo(screen_x, screen_y)

                        # Click when thumb is near index and only thumb, index, and middle fingers are raised
                        if thumb_near_index and not mouse_is_down:
                            print("mouse down")
                            pyautogui.mouseDown()
                            mouse_is_down = True
                        elif not thumb_near_index and mouse_is_down:
                            print("mouse up")
                            pyautogui.mouseUp()
                            mouse_is_down = False

                        # Draw circles on thumb and index finger MCP for visual feedback
                        distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y = (
                            detect_thumb_near_index_mcp(hand_landmarks, height, width)
                        )
                        cv2.circle(
                            frame, (index_mcp_x, index_mcp_y), 10, (0, 255, 255), -1
                        )
                        cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)

                        # Click action when thumb is near index finger MCP
                        if distance < 25:
                            print("mouse click")
                            pyautogui.mouseDown()
                            pyautogui.sleep(0.2)
                            pyautogui.mouseUp()
                            last_click = time()
                    elif mouse_control_active:
                        print("mouse control deactivated")
                        mouse_control_active = False
                        if mouse_is_down:
                            print("mouse up")
                            pyautogui.mouseUp()
                            mouse_is_down = False

                    if current_time - last_scroll_time > scroll_cool_down:
                        if (
                            pinky_tip.y < pinky_pip.y and pinky_tip.y < pinky_mcp.y and pinky_tip.y > index_tip.y and
                             abs(pinky_tip.y - index_tip.y) < 0.05
                        ):
                            print("Switching tabs")
                            pyautogui.keyDown("ctrl")
                            pyautogui.press("tab")
                            pyautogui.keyUp("ctrl")
                            pyautogui.sleep(1)
                            last_switch = time()
                        elif (
                            index_pip.y > index_tip.y
                            and middle_tip.y > middle_mcp.y
                            and (index_pip.y - index_tip.y) * 10 > 1.05
                        ):
                            print("Scrolling up")


                            for _ in range(SCROLL_ITERATIONS):
                                pyautogui.scroll(SCROLL_AMOUNT)

                            pyautogui.scroll(100)


                            for _ in range(SCROLL_ITERATIONS):
                                pyautogui.scroll(SCROLL_AMOUNT)

                            last_scroll_time = current_time
                        elif (
                            index_pip.y < index_tip.y
                            and (index_pip.y - index_tip.y) * 10 < -1.25
                        ):
                            print("Scrolling down")


                            for _ in range(SCROLL_ITERATIONS):
                                pyautogui.scroll(-SCROLL_AMOUNT)
                            last_scroll_time = current_time
                        elif (
                            detect_thumb_near_index_mcp(hand_landmarks, height, width)[
                                0
                            ]
                            > 150
                        ):
                            result = messagebox.askyesno(
                                "Confirm Exit", "Do you actually want to do this?"
                            )
                            if result:
                                sys.exit()

                            pyautogui.scroll(-100)

                            for _ in range(SCROLL_ITERATIONS):
                                pyautogui.scroll(-SCROLL_AMOUNT)

                            last_scroll_time = current_time


    # Display the image
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
        if result:

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
            for hand_landmarks, handedness in zip(process.multi_hand_landmarks, process.multi_handedness):
                label = handedness.classification[0].label

                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                index_mcp_x = index_mcp.x
                index_mcp_y = index_mcp.y

                if frame.shape[1] and frame.shape[0]:
                    if index_mcp_x is not None and index_mcp_y is not None:
                        try:
                            index_mcp_x_px = int(index_mcp_x * frame.shape[1])
                            index_mcp_y_px = int(index_mcp_y * frame.shape[0])

                            cv2.circle(
                                frame,
                                (index_mcp_x_px, index_mcp_y_px),
                                10,
                                (0, 255, 255),
                                -1,
                            )
                        except cv2.error as e:
                            print(f"OpenCV error: {e}")
                    else:
                        raise ValueError(f"Invalid center for circle")

                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if hand_landmarks:
                    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[HandLandmark.PINKY_TIP]

                    volumeup_distance = calculate_distance(thumb_tip, index_tip)
                    volumedown_distance = calculate_distance(thumb_tip, middle_tip)
                    zoom_up_distance = calculate_distance(thumb_tip, pinky_tip)
                    zoom_down_distance = calculate_distance(thumb_tip, pinky_tip)

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
                    prev_prev_x, prev_prev_y = 0, 0
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
                                velocity_x = (screen_x - prev_x) / (current_time - prev_time)
                                velocity_y = (screen_y - prev_y) / (current_time - prev_time)

                                future_x = screen_x + velocity_x * 0.2
                                future_y = screen_y + velocity_y * 0.2

                                pyautogui.moveTo(future_x, future_y)

                            prev_prev_x, prev_prev_y = prev_x, prev_y
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
                                detect_thumb_near_index_mcp(hand_landmarks, height, width)
                            )
                            cv2.circle(frame, (index_mcp_x, index_mcp_y), 10, (0, 255, 255), -1)
                            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)

                            if distance < 25:
                                pyautogui.mouseDown()
                                pyautogui.sleep(0.2)
                                pyautogui.mouseUp()
                                last_click = time()
                        elif mouse_control_active:
                            mouse_control_active = False
                            if mouse_is_down:
                                pyautogui.mouseUp()
                                mouse_is_down = False

                        if current_time - last_scroll_time > scroll_cool_down:
                            if pinky_tip.y < middle_tip.y and abs(pinky_tip.y - index_tip.y) < 0.05:
                                pyautogui.hotkey("ctrl", "tab")
                                pyautogui.sleep(1)
                                last_switch = time()
                            elif (middle_tip.y > middle_tip.y) and (index_tip.y - middle_tip.y) * 10 > 1.05:
                                pyautogui.scroll(100)
                                last_scroll_time = current_time
                            elif (middle_tip.y < middle_tip.y) and (middle_tip.y - index_tip.y) * 10 < -1.25:
                                pyautogui.scroll(-100)
                                last_scroll_time = current_time

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
            if result:
                stop_thread = True
                break

    cap.release()
    cv2.destroyAllWindows()

# Create and start the capture thread
capture_thread = threading.Thread(target=capture_and_process)
capture_thread.start()

# Main thread can handle other tasks or UI interactions here

# Wait for the capture thread to finish before exiting the script
capture_thread.join()

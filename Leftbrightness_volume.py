from time import time
from math import hypot

import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import HandLandmark
import numpy as np
import pyautogui
from tkinter import messagebox

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

prev_x, prev_y = 0.0, 0.0
last_scroll_time = time()
scroll_cool_down = 1
min_scroll_distance = 20


def calculate_distance(point1, point2):
    return hypot(point2.x - point1.x, point2.y - point1.y)


def detect_two_fingers_up(hand_landmarks):
    # Check if index and middle fingers are up
    index_tip = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[HandLandmark.MIDDLE_FINGER_MCP]

    return index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y


def detect_thumb_near_index_mcp(hand_landmarks, height, width):
    # Calculate the distance between thumb tip and index finger MCP
    index_mcp = hand_landmarks.landmark[HandLandmark.INDEX_FINGER_MCP]
    thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]

    index_mcp_x = int(index_mcp.x * width)
    index_mcp_y = int(index_mcp.y * height)
    thumb_x = int(thumb_tip.x * width)
    thumb_y = int(thumb_tip.y * height)

    distance = np.sqrt((index_mcp_x - thumb_x) ** 2 + (index_mcp_y - thumb_y) ** 2)
    return distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y


# Starting webcam to capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

                # Right hand gestures
                if label == "Right":
                    current_time = time()
                    if detect_two_fingers_up(hand_landmarks):
                        print("mouse control")
                        # Normalize the coordinates within the frame
                        norm_x = (middle_tip.x - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)
                        norm_y = (middle_tip.y - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)

                        # Clamp normalized coordinates to [0, 1] range
                        norm_x = min(max(norm_x, 0), 1)
                        norm_y = min(max(norm_y, 0), 1)

                        # Convert normalized coordinates to screen coordinates
                        screen_width, screen_height = pyautogui.size()
                        screen_x = int(screen_width * norm_x)
                        screen_y = int(screen_height * norm_y)

                        # Smooth the cursor movement
                        smooth_x = prev_x + (screen_x - prev_x) * SMOTH_FACTOR
                        smooth_y = prev_y + (screen_y - prev_y) * SMOTH_FACTOR
                        prev_x, prev_y = smooth_x, smooth_y

                        # Move the mouse pointer
                        pyautogui.moveTo(smooth_x, smooth_y)

                        # Optionally draw a circle at the cursor position on the image
                        cursor_x = int(width * middle_tip.x)
                        cursor_y = int(height * middle_tip.y)
                        cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)

                        # Detect thumb near index finger MCP for clicking
                        distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y = (
                            detect_thumb_near_index_mcp(hand_landmarks, height, width)
                        )

                        # draw circles on thumb and index finger MCP for visual feedback
                        cv2.circle(
                            frame, (index_mcp_x, index_mcp_y), 10, (0, 255, 255), -1
                        )
                        cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)

                        # Click action when thumb is near index finger MCP
                        if distance < 40:
                            print("mouse click")
                            pyautogui.click()
                            pyautogui.sleep(0.1)

                    if current_time - last_scroll_time > scroll_cool_down:
                        if (
                            pinky_tip.y < pinky_pip.y
                            and abs(pinky_tip.y - index_tip.y) < 0.05
                        ):
                            print("Switching tabs")
                            pyautogui.hotkey("ctrl", "tab")
                            pyautogui.sleep(0.1)
                            last_switch = time()
                        elif (
                            index_pip.y > index_tip.y
                            and middle_tip.y > middle_mcp.y
                            and (index_pip.y - index_tip.y) * 10 > 1.05
                        ):
                            print("Scrolling up")
                            pyautogui.scroll(380)
                            last_scroll_time = current_time
                        elif (
                            index_pip.y < index_tip.y
                            and (index_pip.y - index_tip.y) * 10 < -1.25
                        ):
                            print("Scrolling down")
                            pyautogui.scroll(-50)
                            last_scroll_time = current_time

                    # not comfortable with this gesture
                    # if (
                    #     thumb_tip.y > thumb_base.y
                    #     and index_tip.y < thumb_tip.y
                    #     and index_tip.y < index_pip.y
                    # ):
                    #     print("Turning off webcam")
                    #     result = messagebox.askyesno(
                    #         "Confirm Exit", "Do you actually want to do this?"
                    #     )
                    #
                    #     if result:
                    #         sys.exit(1)

    # Display the image
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
        if result:
            break

cap.release()
cv2.destroyAllWindows()

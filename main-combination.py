import cv2
import time
import mediapipe as mp
import pyautogui
import numpy as np
from math import hypot
from tkinter import messagebox

pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define region of interest (ROI) within the webcam frame
ROI_TOP = 0.2  # Top 20% of the frame height
ROI_BOTTOM = 0.8  # Bottom 80% of the frame height
ROI_LEFT = 0.2  # Left 20% of the frame width
ROI_RIGHT = 0.8  # Right 80% of the frame width

# Smoothing parameters
prev_x, prev_y = 0.0, 0.0
smooth_factor = 0.2

# Scrolling parameters
last_index_y = None
last_scroll_time = time.time()
scroll_cooldown = 1  # Cooldown period for scroll actions in seconds
min_scroll_distance = 20  # Minimum movement for scroll action

def detect_two_fingers_up(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    return index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y

def detect_thumb_near_index_mcp(hand_landmarks, height, width):
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp_x = int(index_mcp.x * width)
    index_mcp_y = int(index_mcp.y * height)
    thumb_x = int(thumb_tip.x * width)
    thumb_y = int(thumb_tip.y * height)
    distance = np.sqrt((index_mcp_x - thumb_x) ** 2 + (index_mcp_y - thumb_y) ** 2)
    return distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y

def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    height, width, _ = image.shape
    top_left = (int(width * ROI_LEFT), int(height * ROI_TOP))
    bottom_right = (int(width * ROI_RIGHT), int(height * ROI_BOTTOM))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmark_list = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                landmark_list.append([idx, int(lm.x * width), int(lm.y * height)])

            if label == 'Left':
                thumb_tip = landmark_list[4]
                index_tip = landmark_list[8]
                middle_tip = landmark_list[12]
                ring_tip = landmark_list[16]
                pinky_tip = landmark_list[20]

                volumeup_distance = calculate_distance(thumb_tip[1:], index_tip[1:])
                volumedown_distance = calculate_distance(thumb_tip[1:], middle_tip[1:])
                zoom_up_distance = calculate_distance(thumb_tip[1:], ring_tip[1:])
                zoom_down_distance = calculate_distance(thumb_tip[1:], pinky_tip[1:])

                if volumeup_distance < 40:
                    pyautogui.press("volumeup")
                if volumedown_distance < 40:
                    pyautogui.press("volumedown")
                if zoom_up_distance < 40:
                    pyautogui.hotkey("ctrl", "+")
                if zoom_down_distance < 40:
                    pyautogui.hotkey("ctrl", "-")

            if label == 'Right':
                thumb_base = landmark_list[2]
                thumb_tip = landmark_list[4]
                index_pip = landmark_list[6]
                index_tip = landmark_list[8]
                middle_pip = landmark_list[10]
                middle_tip = landmark_list[12]
                ring_pip = landmark_list[14]
                ring_tip = landmark_list[16]
                pinky_pip = landmark_list[18]
                pinky_tip = landmark_list[20]

                if last_index_y is None:
                    last_index_y = index_tip[2]

                current_time = time.time()
                if current_time - last_scroll_time > scroll_cooldown:
                    if index_tip[2] < middle_tip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance:
                        pyautogui.scroll(380)
                        last_scroll_time = current_time
                    elif thumb_tip[2] < index_tip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance:
                        pyautogui.scroll(-50)
                        last_scroll_time = current_time
                    elif pinky_tip[2] < pinky_pip[2] and pinky_tip[2] > index_tip[2]:
                        pyautogui.keyDown("ctrl")
                        pyautogui.press("tab")
                        pyautogui.keyUp("ctrl")
                        time.sleep(1)
                
                last_index_y = index_tip[2]

                if detect_two_fingers_up(hand_landmarks):
                    norm_x = (index_tip[1] - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)
                    norm_y = (index_tip[2] - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)
                    norm_x = min(max(norm_x, 0), 1)
                    norm_y = min(max(norm_y, 0), 1)
                    screen_width, screen_height = pyautogui.size()
                    screen_x = int(screen_width * norm_x)
                    screen_y = int(screen_height * norm_y)
                    smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
                    smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
                    prev_x, prev_y = smooth_x, smooth_y
                    pyautogui.moveTo(smooth_x, smooth_y)
                    cursor_x = int(width * index_tip[1])
                    cursor_y = int(height * index_tip[2])
                    cv2.circle(image, (cursor_x, cursor_y), 10, (0, 255, 0), -1)
                    roi_cursor_x = int(top_left[0] + norm_x * (bottom_right[0] - top_left[0]))
                    roi_cursor_y = int(top_left[1] + norm_y * (bottom_right[1] - top_left[1]))
                    cv2.circle(image, (roi_cursor_x, roi_cursor_y), 10, (0, 0, 255), -1)
                    cv2.putText(image, f'({int(smooth_x)}, {int(smooth_y)})', (cursor_x, cursor_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y = detect_thumb_near_index_mcp(hand_landmarks, height, width)
                cv2.circle(image, (index_mcp_x, index_mcp_y), 10, (0, 255, 255), -1)
                cv2.circle(image, (thumb_x, thumb_y), 10, (0, 255, 255), -1)
                if distance < 40:
                    pyautogui.click()
                    pyautogui.sleep(0.1)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
        if result:
            break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define region of interest (ROI) within the webcam frame
ROI_TOP = 0.2  # Top 20% of the frame height
ROI_BOTTOM = 0.8  # Bottom 80% of the frame height
ROI_LEFT = 0.2  # Left 20% of the frame width
ROI_RIGHT = 0.8  # Right 80% of the frame width

# Smoothing parameters
prev_x, prev_y = 0, 0
smooth_factor = 0.2

def detect_two_fingers_up(hand_landmarks):
    # Check if index and middle fingers are up
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    if index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y:
        return True
    return False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw ROI border
    height, width, _ = image.shape
    top_left = (int(width * ROI_LEFT), int(height * ROI_TOP))
    bottom_right = (int(width * ROI_RIGHT), int(height * ROI_BOTTOM))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            if label == 'Right':
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if detect_two_fingers_up(hand_landmarks):
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Normalize the coordinates within the frame
                    norm_x = (index_tip.x - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)
                    norm_y = (index_tip.y - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)

                    # Clamp normalized coordinates to [0, 1] range
                    norm_x = min(max(norm_x, 0), 1)
                    norm_y = min(max(norm_y, 0), 1)

                    # Convert normalized coordinates to screen coordinates
                    screen_width, screen_height = pyautogui.size()
                    screen_x = int(screen_width * norm_x)
                    screen_y = int(screen_height * norm_y)

                    # Smooth the cursor movement
                    smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
                    smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
                    prev_x, prev_y = smooth_x, smooth_y

                    # Move the mouse pointer
                    pyautogui.moveTo(smooth_x, smooth_y)

                    # Draw a circle at the cursor position on the image
                    cursor_x = int(width * index_tip.x)
                    cursor_y = int(height * index_tip.y)
                    cv2.circle(image, (cursor_x, cursor_y), 10, (0, 255, 0), -1)

                    # Draw a dot within the ROI to indicate the finger position
                    roi_cursor_x = int(top_left[0] + norm_x * (bottom_right[0] - top_left[0]))
                    roi_cursor_y = int(top_left[1] + norm_y * (bottom_right[1] - top_left[1]))
                    cv2.circle(image, (roi_cursor_x, roi_cursor_y), 10, (0, 0, 255), -1)

                    # Optionally, display the coordinates on the image
                    cv2.putText(image, f'({int(smooth_x)}, {int(smooth_y)})', (cursor_x, cursor_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', image)

    # Adjust the key to something else if needed
    if cv2.waitKey(5) & 0xFF == 27:  # Use the 'Esc' key to close the program
        break

cap.release()
cv2.destroyAllWindows()
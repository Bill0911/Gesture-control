import cv2
import time
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set a smaller frame size for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define region of interest (ROI) within the webcam frame
ROI_TOP = 0.2  # Top 20% of the frame height
ROI_BOTTOM = 0.8  # Bottom 80% of the frame height
ROI_LEFT = 0.2  # Left 20% of the frame width
ROI_RIGHT = 0.8  # Right 80% of the frame width

# Smoothing parameters
prev_x, prev_y = 0, 0
smooth_factor = 0.8

# Dragging state
is_dragging = False

def detect_two_fingers_up(hand_landmarks):
    # Check if index and middle fingers are up
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    return index_tip.y < index_mcp.y and middle_tip.y < middle_mcp.y

def detect_thumb_near_index_mcp(hand_landmarks, height, width):
    # Calculate the distance between thumb tip and index finger MCP
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    index_mcp_x = int(index_mcp.x * width)
    index_mcp_y = int(index_mcp.y * height)
    thumb_x = int(thumb_tip.x * width)
    thumb_y = int(thumb_tip.y * height)

    distance = np.sqrt((index_mcp_x - thumb_x) ** 2 + (index_mcp_y - thumb_y) ** 2)
    return distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y

while cap.isOpened():
    start_time = time.time()
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
                if detect_two_fingers_up(hand_landmarks):
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    
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
                    smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
                    smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
                    prev_x, prev_y = smooth_x, smooth_y

                    # Move the mouse pointer
                    pyautogui.moveTo(smooth_x, smooth_y)

                    # Optionally draw a circle at the cursor position on the image
                    cursor_x = int(width * middle_tip.x)
                    cursor_y = int(height * middle_tip.y)
                    cv2.circle(image, (cursor_x, cursor_y), 10, (0, 255, 0), -1)
                
                # Detect thumb near index finger MCP for clicking
                distance, index_mcp_x, index_mcp_y, thumb_x, thumb_y = detect_thumb_near_index_mcp(hand_landmarks, height, width)
                
                # Draw circles on thumb and index finger MCP for visual feedback
                cv2.circle(image, (index_mcp_x, index_mcp_y), 10, (0, 255, 255), -1)
                cv2.circle(image, (thumb_x, thumb_y), 10, (0, 255, 255), -1)
                
                # Click and drag functionality
                if distance < 40:
                    if not is_dragging:
                        pyautogui.mouseDown()
                        is_dragging = True
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

    # Display the frame
    cv2.imshow('Hand Tracking', image)

    # Break the loop on 'Esc' key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps}")

cap.release()
cv2.destroyAllWindows()
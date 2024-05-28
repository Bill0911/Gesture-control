import cv2
import time
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
import pyautogui
from tkinter import messagebox

# Creating the hand recognizer model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2,
)

Draw = mp.solutions.drawing_utils


# Define region of interest (ROI) within the webcam frame
ROI_TOP = 0.2  # Top 20% of the frame height
ROI_BOTTOM = 0.8  # Bottom 80% of the frame height
ROI_LEFT = 0.2  # Left 20% of the frame width
ROI_RIGHT = 0.8  # Right 80% of the frame width

# Smoothing parameters
prev_x, prev_y = 0.0, 0.0
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


# Starting webcam to capture
cap = cv2.VideoCapture(0)

pyautogui.FAILSAFE = False


def calculate_distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])


last_index_y = None
last_scroll_time = time.time()
scroll_coolDown = 1  # Cooldown period for scroll actions in seconds
min_scroll_distance = 20  # Minimum movement for scroll action

while True:
    # Read video frame
    ret, frame = cap.read()

    if not ret:
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB image
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processing RGB image
    process = hands.process(frameRGB)

    height, width, _ = frame.shape
    top_left = (int(width * ROI_LEFT), int(height * ROI_TOP))
    bottom_right = (int(width * ROI_RIGHT), int(height * ROI_BOTTOM))

    landmarkList = []
    if process.multi_hand_landmarks:
        for hand_no, handlm in enumerate(process.multi_hand_landmarks):
            # Identify if the hand is left or right
            handedness = process.multi_handedness[hand_no].classification[0].label

            for _id, lm in enumerate(handlm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarkList.append([_id, cx, cy])

            # Draw landmarks
            Draw.draw_landmarks(frame, handlm, mp_hands.HAND_CONNECTIONS)

            if landmarkList:
                # Get coordinates for fingers
                thumb_base = landmarkList[2]
                thumb_tip = landmarkList[4]
                index_pip = landmarkList[6]
                index_tip = landmarkList[8]
                middle_pip = landmarkList[10]
                middle_tip = landmarkList[12]
                ring_pip = landmarkList[14]
                ring_tip = landmarkList[16]
                pinky_pip = landmarkList[18]
                pinky_tip = landmarkList[20]

                if last_index_y is None:
                    last_index_y = index_tip[2]

                # Calculate distances
                volumeup_distance = calculate_distance(thumb_tip, index_tip)
                volumedown_distance = calculate_distance(thumb_tip, middle_tip)
                zoom_up_distance = calculate_distance(thumb_tip, ring_tip)
                zoom_down_distance = calculate_distance(thumb_tip, pinky_tip)

                # Left hand gestures
                if handedness == "Left":
                    # Adjust volume for left hand
                    volumeup_level = np.interp(volumeup_distance, [0, 220], [0, 100])
                    if volumeup_level < 10:
                        pyautogui.press("volumeup")
                    volumedown_level = np.interp(
                        volumedown_distance, [0, 220], [0, 100]
                    )
                    if volumedown_level < 10:
                        pyautogui.press("volumedown")

                    # Adjust zoom for left hand
                    zoom_up_level = np.interp(zoom_up_distance, [0, 220], [0, 100])
                    if zoom_up_level < 10:
                        pyautogui.hotkey("ctrl", "+")
                    zoom_down_level = np.interp(zoom_down_distance, [0, 220], [0, 100])
                    if zoom_down_level < 10:
                        pyautogui.hotkey("ctrl", "-")

                # Right hand gestures
                elif handedness == "Right":
                    current_time = time.time()
                    if current_time - last_scroll_time > scroll_coolDown:
                        if (
                            index_tip[2] < middle_tip[2]
                            and abs(index_tip[2] - last_index_y) > min_scroll_distance
                        ):
                            print("Scrolling up")
                            pyautogui.scroll(380)
                            last_scroll_time = current_time
                        elif (
                            thumb_tip[2] < index_tip[2]
                            and abs(index_tip[2] - last_index_y) > min_scroll_distance
                        ):
                            print("Scrolling down")
                            pyautogui.scroll(-50)
                            last_scroll_time = current_time
                        elif (
                            pinky_tip[2] < pinky_pip[2] and pinky_tip[2] > index_tip[2]
                        ):
                            print("Switching tabs")
                            pyautogui.keyDown("ctrl")
                            pyautogui.press("tab")
                            pyautogui.keyUp("ctrl")
                            time.sleep(1)
                            last_switch = time.time()

                    if detect_two_fingers_up(handlm):
                        print("mouse control")

                        # Normalize the coordinates within the frame
                        norm_x = (index_tip[1] - ROI_LEFT) / (ROI_RIGHT - ROI_LEFT)
                        norm_y = (index_tip[1] - ROI_TOP) / (ROI_BOTTOM - ROI_TOP)

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
                        cursor_x = int(width * index_tip[2])
                        cursor_y = int(height * index_tip[1])
                        cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 0), -1)

                        # Draw a dot within the ROI to indicate the finger position
                        roi_cursor_x = int(
                            top_left[0] + norm_x * (bottom_right[0] - top_left[0])
                        )
                        roi_cursor_y = int(
                            top_left[1] + norm_y * (bottom_right[1] - top_left[1])
                        )
                        cv2.circle(
                            frame, (roi_cursor_x, roi_cursor_y), 10, (0, 0, 255), -1
                        )

                        # Optionally, display the coordinates on the image
                        cv2.putText(
                            frame,
                            f"({int(smooth_x)}, {int(smooth_y)})",
                            (cursor_x, cursor_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                    elif (
                        thumb_tip[2] > thumb_base[2]
                        and index_tip[2] < thumb_tip[2]
                        and index_tip[2] < index_pip[2]
                    ):
                        print("Turning off webcam")
                        result = messagebox.askyesno(
                            "Confirm Exit", "Do you actually want to do this?"
                        )
                        if result:
                            break

    # Display the image
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        result = messagebox.askyesno("Confirm Exit", "Do you actually want to do this?")
        if result:
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import time
import pyautogui
import mediapipe as mp

# Import MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Setup the webcam
cap = cv2.VideoCapture(0)

# Set the time of the last tab switch
last_switch = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the hand landmarks
    results = hands.process(image)

    # Check if any hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmarks for the index finger tip and PIP
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            # Check if the index finger is raised and at least 0.5 seconds have passed since the last tab switch
            if index_pip.y - index_tip.y > 0.1 and time.time() - last_switch > 0.5:
                print("Switching tabs")  # Debugging line
                # Press down 'ctrl'
                pyautogui.keyDown('ctrl')
                # Press 'tab'
                pyautogui.press('tab')
                # Release 'ctrl'
                pyautogui.keyUp('ctrl')
                # Update the time of the last tab switch
                last_switch = time.time()
            else: #Test if the switching tabs feature is stable or not 
                print("Not switching tabs")  # Debugging line
                print(f"index_tip.y: {index_tip.y}, index_pip.y: {index_pip.y}, time since last switch: {time.time() - last_switch}")

    # Show the image
    cv2.imshow('Image', frame)

    # press 'q' to turn off the feature
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
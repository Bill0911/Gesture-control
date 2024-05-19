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
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            # Check if the index finger is above the middle finger
        if index_tip.y < middle_tip.y:
            print("Scrolling up")  # Debugging line
            pyautogui.scroll(50)  # Scroll up

            # Check if the middle finger is above the index finger
        elif middle_tip.y < index_tip.y:
            print("Scrolling down")  # Debugging line
            pyautogui.scroll(-50)  # Scroll down

    # Show the image
    cv2.imshow('Image', frame)

    # press 'q' to turn off the feature
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

# Get default audio device using PyCaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get current volume
current_volume_db = volume.GetMasterVolumeLevel()

# Get volume range
volume_range = volume.GetVolumeRange()
min_volume_db = volume_range[0]
max_volume_db = volume_range[1]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the Euclidean distance between the thumb tip and the index finger tip
            distance = np.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)

            # Change the volume based on the distance
            if distance < 0.1:  # Decrease the volume when fingers are touching
                new_volume_db = max(min_volume_db, current_volume_db - 6.0)
                volume.SetMasterVolumeLevel(new_volume_db, None)
                color = (0, 0, 255)  # Red
            elif distance > 0.2:  # Lower the gap required to increase volume
                new_volume_db = min(max_volume_db, current_volume_db + 6.0)
                volume.SetMasterVolumeLevel(new_volume_db, None)
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue

            # Update current volume
            current_volume_db = volume.GetMasterVolumeLevel()

            # Draw a line between the thumb tip and the index finger tip
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            index_finger_tip_coords = (
            int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))

            cv2.line(frame, thumb_tip_coords, index_finger_tip_coords, color, 2)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
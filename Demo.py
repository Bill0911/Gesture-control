import cv2 
import mediapipe as mp 
from math import hypot 
import screen_brightness_control as sbc 
import numpy as np
import pyautogui
import time

# creating the hand recognizer model 
mpHands = mp.solutions.hands
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2)

Draw = mp.solutions.drawing_utils 

# Starting webcam to capture 
cap = cv2.VideoCapture(0) 

#For scroll up-down feature.
#Initialize variables
last_scroll_time = 0
scroll_coolDown = 0.5 
min_scroll_distance = 0.05 
last_index_y = None

#For brightness adjustment feature.
#Initialize variables
last_brightness_time = None
brightness_adjustment_coolDown = 3

while True: 
    # Read video frame 
    _, frame = cap.read() 

    # Flip image 
    frame = cv2.flip(frame, 1) 

    # Convert BGR image to red-green-blue image (OpenCV) 
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    # Processing red-green-blue image 
    Process = hands.process(frameRGB)

    landmarkList = [] 
    # if hands are present in image(by frame) 
    if Process.multi_hand_landmarks: 
        # recognize handmarks 
        for handlm in Process.multi_hand_landmarks: 
            for _id, landmarks in enumerate(handlm.landmark): 
                # store height and width of image 
                height, width, color_channels = frame.shape 

                # calculate & append x, y coordinates 
                # of handmarks from image(frame) to lmList 
                x, y = int(landmarks.x*width), int(landmarks.y*height) 
                landmarkList.append([_id, x, y]) 

            # draw landmarks 
            Draw.draw_landmarks(frame, handlm, 
                                mpHands.HAND_CONNECTIONS) 

    # If landmarks list !empty() 
    if landmarkList != []: 
        # store x,y coordinates of tip of thumb 
        x_1, y_1 = landmarkList[4][1], landmarkList[4][2] 

        # store x,y coordinates of tip of index finger 
        x_2, y_2 = landmarkList[8][1], landmarkList[8][2] 

        # draw circle on thumb and index finger tip 
        cv2.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED) 
        cv2.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED) 

        # draw line from tip of thumb to tip of index finger 
        cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3) 

        # calculate square root of the sum of 
        # squares of the specified arguments. 
        L = hypot(x_2-x_1, y_2-y_1) 

        # 1-D linear interpolant to a function 
        # with given discrete data points 
        # (Hand range 15 - 220, Brightness 
        # range 0 - 100), evaluated at length. 
        b_level = np.interp(L, [15, 220], [0, 100]) 
     
    # set up brightness
     
        if last_brightness_time is None:
            last_brightness_time = time.time()
        elif time.time() - last_brightness_time:
            sbc.set_brightness(int(b_level))
            last_brightness_time = time.time()
        
        #Scrolling
        if len(landmarkList) >= 21:  # Make sure the index finger tip and middle finger tip are detected
            index_tip = landmarkList[8]
            middle_tip = landmarkList[12]
            index_pip = landmarkList[6]
            pinky_tip = landmarkList[20]
            pinky_pip = landmarkList[18]

            if last_index_y is None:
                last_index_y = index_tip[2]
    
            if time.time() - last_scroll_time > scroll_coolDown:
                if index_tip[2] < middle_tip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance:
                    print("Scrolling up")
                    pyautogui.scroll(90)
                    last_scroll_time = time.time()
            elif index_tip[2] < pinky_tip[2] and pinky_tip[2] < pinky_pip[2] and abs(index_tip[2] - last_index_y) > min_scroll_distance:
                    print("Scrolling down")
                    pyautogui.scroll(-50)
                    last_scroll_time = time.time()

        last_index_y = index_tip[2]
            
    #If the user wants to turn off the webcam, press 'q'
    cv2.imshow('Image', frame) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break
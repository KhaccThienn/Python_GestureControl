import cv2
import mediapipe as mp

import math
import numpy as np
import time

import pyautogui

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import screen_brightness_control


class GestureRecognizer():
    """
    A class for gesture recognition and control.

    Explanation:
    - Provides methods to extract landmarks, control mouse movements, adjust volume, brightness, and perform Alt+Tab actions based on hand gestures.
    - Utilizes hand landmarks to interpret gestures and trigger corresponding actions.

    """
    
    def __init__(self):
        """
        Initializes the GestureRecognizer class with necessary components for hand gesture recognition.

        Explanation:
        - Initializes the GestureRecognizer with components for hand gesture recognition.
        - Sets up the required objects for processing hand gestures.
        """
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.drawTools = mp.solutions.drawing_utils
        
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        
    
    def getLandmarks(self, img, draw = False):
        """
        Extracts landmarks from an image and optionally draws them.

        Explanation:
        - Processes the input image to extract landmarks of detected hands.
        - Optionally draws the landmarks on the image if specified.

        Args:
            img: The input image to extract landmarks from.
            draw: A boolean flag to specify if landmarks should be drawn on the image.

        Returns:
            A list of landmarks and the image with or without drawn landmarks.
        """    
        landmarks = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:

                for id,lm in enumerate(handLandmarks.landmark):
                    height, width, channels = img.shape

                    # Current x and y coordinates
                    cx, cy = int( lm.x * width ), int( lm.y * height )

                    landmarks.append([id,cx,cy])

                # draw landmark skeleton
                if draw:
                    self.drawTools.draw_landmarks(img,handLandmarks, self.mpHands.HAND_CONNECTIONS)
        
        return landmarks, img


    def fingerUp(self, img, landmarks, isLeftHand, draw = False):
        """
        Determines the fingers that are up based on landmark positions and hand orientation.

        Explanation:
        - Identifies the fingers that are raised by analyzing the landmark positions and hand orientation.
        - Returns a list indicating the status of each finger and optionally displays the count on the image.

        Args:
            img: The input image containing the hand landmarks.
            landmarks: List of landmark coordinates.
            isLeftHand: A boolean flag indicating if the hand is the left hand.
            draw: A boolean flag to specify if the finger count should be displayed on the image.

        Returns:
            A list representing the status of each finger and the image with optional finger count displayed.
        """
        fingers = []
        tipIds = [8,12,16,20]       # Landmarks of the index, middle, ring and pinky finger
        count = 0

        # Left Hand Thumb
        if isLeftHand:
            if landmarks[4][1] < landmarks[3][1]:
                count += 1
                fingers.append(1)
            else:
                fingers.append(0)
        elif landmarks[4][1] > landmarks[3][1]:
            count += 1
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers (orientation is the same for both left and right hands)
        # id-2 is the id for the respective finger's middle groove
        for id in tipIds:
            if landmarks[id][2] > landmarks[id-2][2]:
                fingers.append(0)

            else:
                fingers.append(1)
                count += 1

        # show number of fingers that are up
        if draw:
            cv2.putText(img, str(count), (50,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

        return fingers, img
    
    def findDistance(self, lm1, lm2, img, landmarks, draw = False ):
        """
        Calculates the Euclidean distance between two landmarks and optionally visualizes it on the image.

        Explanation:
        - Computes the distance between two specified landmarks based on their coordinates.
        - Optionally draws a line connecting the landmarks and circles around each landmark on the image.

        Args:
            lm1: Index of the first landmark in the landmarks list.
            lm2: Index of the second landmark in the landmarks list.
            img: The input image containing the hand landmarks.
            landmarks: List of landmark coordinates.
            draw: A boolean flag to specify if the distance line and circles should be drawn on the image.

        Returns:
            The Euclidean distance between the two landmarks and the image with optional visualizations.
        """
        x1,y1 = landmarks[lm1][1:]
        x2,y2 = landmarks[lm2][1:]

        # Euclidean distance between 2 points
        length = math.hypot(x2-x1, y2-y1) 

        # Display line between the two landmarks
        if draw:
            cv2.line(img, (x1,y1),(x2,y2), (255,0,255), 3 )
            cv2.circle(img, (x1,y1),5,(255,0,0), cv2.FILLED)
            cv2.circle(img, (x2,y2),5,(255,0,0), cv2.FILLED)

        return length, img
    
    def isLeftHand(self, img):
        """
        Determines if the hand in the image is the left hand.

        Explanation:
        - Processes the image to identify the orientation of the hand as left or right.
        - Returns a tuple indicating if a hand is present and if it is the left hand.

        Args:
            img: The input image containing the hand to be analyzed.

        Returns:
            A tuple (hand_present, is_left_hand) where hand_present is a boolean indicating if a hand is detected,
            and is_left_hand is a boolean indicating if the detected hand is the left hand.
        """
        # Mirror the current image
        flipped_image = cv2.flip(img, 1)
        orientation_results = self.hands.process(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))

        handType = orientation_results.multi_handedness
        whichHand = None

        if handType:
            whichHand = handType[0].classification[0].label
            return (True, True) if whichHand == "Left" else (True, False)
        else:
            return False, False

    def scrollControl(self ,fingers, scrollSpeed = 50):
        """
        Controls scrolling based on the status of fingers.

        Explanation:
        - Scrolls up or down based on the configuration of fingers.
        - Adjusts the scroll speed according to the specified value.

        Args:
            fingers: A list representing the status of each finger (0 for down, 1 for up).
            scrollSpeed: The speed at which scrolling should occur (default is 50).

        Returns:
            None. Scrolls the content up or down based on finger configuration.
        """
        if fingers == [0,0,0,0,0]:
            pyautogui.scroll(-scrollSpeed)
            print("Scrolling Down")
        elif fingers == [1,1,1,1,1]:
            pyautogui.scroll(scrollSpeed)
            print("Scrolling Up")

    def mouseControl(self, img, landmarks, fingers, draw = True):
        """
        Controls the mouse based on hand gestures.

        Explanation:
        - Maps hand gestures to mouse movements and actions.
        - Moves the mouse cursor and performs left-click based on finger positions and gestures.

        Args:
            img: The input image containing the hand and control areas.
            landmarks: List of landmark coordinates.
            fingers: A list representing the status of each finger for gesture recognition.
            draw: A boolean flag to specify if visualizations should be drawn on the image.

        Returns:
            None. Controls the mouse movements and actions based on hand gestures.
        """
        
        # Bounding Rectangle (Mouse Control Area)
        frameR_x = 70
        frameR_y = 50

        # Neutral Area
        inner_x = frameR_x + 125
        inner_y = frameR_y + 70

        if fingers == [0,1,0,0,0]:

            # Coords of tip of index finger
            x1,y1 = landmarks[8][1:]

            relative_x = 40
            relative_y = 30            

            frame_width = 400
            frame_height = 250

            inner_width = 150
            inner_height = 100

            # Better control if the boundaries are visible
            if draw:
                # Mouse Control Bounding rectangle
                cv2.rectangle(img, (frameR_x, frameR_y), (frameR_x + frame_width, frameR_y + frame_height), (255,0,255), 3)

                # No-Movement Zone / Neutral Area
                cv2.rectangle(img, (inner_x, inner_y), (inner_x + inner_width, inner_y + inner_height), (255,0,0), 3)

                cv2.circle(img,(x1,y1),10, (255,0,255), cv2.FILLED)

            # Move mouse to the left
            if x1 >= inner_x + inner_width and x1 <= frameR_x + frame_width:
                pyautogui.move(-relative_x, 0) 
                print("Mouse -> left")

            # Move mouse to the right
            elif x1 <= inner_x and x1 >= frameR_x:
                pyautogui.move(relative_x, 0)  
                print("Mouse -> right")

            # Move mouse downwards
            elif y1 <= inner_y and y1 >= frameR_y:
                pyautogui.move(0, -relative_y)  
                print("Mouse -> down")

            # Move mouse upwards
            elif y1 <= frameR_y + frame_height and y1 >= inner_y + inner_height:
                pyautogui.move(0, relative_y)  
                print("Mouse -> up")

        # If the index finger is up, and the thumb gets stretched out, do left click
        if fingers == [1,1,0,0,0]:
            pyautogui.click()
            print("Left Click!")

    def volumeControl(self, img, landmarks, draw=True):
        """
        Controls the system volume based on hand gestures.

        Explanation:
        - Adjusts the system volume based on the distance between specific hand landmarks.
        - Visualizes the volume level and changes the volume if a significant difference is detected.

        Args:
            img: The input image containing the hand landmarks and volume control visuals.
            landmarks: List of landmark coordinates.
            draw: A boolean flag to specify if volume control visuals should be displayed on the image.

        Returns:
            None. Manages the system volume based on hand gestures and visualizes the volume level.
        """
        # Taking the distance between landmarks 8 (tip of the index) and 4 (tip of the thumb)
        length, img = self.findDistance(4,8,img,landmarks,draw=True)

        # The distance between the fingers ranges from 30 to 200
        new_vol = np.interp(length, (30,200), (0,1))

        currentVolume = self.volume.GetMasterVolumeLevelScalar()

        # For a volume change to be valid there should be at least 10% difference. Done to retain smooth functioning
        threshold = 0.1

        volume_percent = np.interp(currentVolume, (0,1), (0,100) )
        volume_bar = np.interp(currentVolume, (0,1), (0,200))

        # Visual Volume Bar
        if draw:
            cv2.putText(
                img,
                f"{int(volume_percent)}%",
                (50, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 0, 0),
                3,
            )
            cv2.rectangle(img,(50,120),(80,320), (255,0,0), 3)
            cv2.rectangle(img,(50,320-int(volume_bar)),(80,320), (255,0,0), cv2.FILLED)

        # If the difference between the new volume and the old one is greater than the threshold, then change the volume
        if abs(currentVolume - new_vol) >= threshold:
            self.volume.SetMasterVolumeLevelScalar(new_vol, None)
            print("Volume Changed")

    def brightnessControl(self, img, landmarks, draw = False):
        """
        Controls the screen brightness based on hand gestures.

        Explanation:
        - Adjusts the screen brightness based on the distance between specific hand landmarks.
        - Visualizes the brightness level and changes the brightness if a significant difference is detected.

        Args:
            img: The input image containing the hand landmarks and brightness control visuals.
            landmarks: List of landmark coordinates.
            draw: A boolean flag to specify if brightness control visuals should be displayed on the image.

        Returns:
            None. Manages the screen brightness based on hand gestures and visualizes the brightness level.
        """
        
        # Taking the distance between landmarks 8 (tip of the index) and 4 (tip of the thumb)
        length, img = self.findDistance(4,8,img,landmarks,draw=True)

        # The distance between the fingers ranges from 30 to 200
        new_brightness = np.interp(length, (30,200), (0,100))

        currentBrightness = screen_brightness_control.get_brightness()

        # For a brightness change to be valid there should be at least 10% difference. Done to retain smooth functioning
        threshold = 10

        brightness_percent = np.interp(currentBrightness, (0,100), (0,100) )
        brightness_bar = np.interp(currentBrightness, (0,100), (0,200))

        colorOrange = (102,178,255)

        # Visual Brightness Bar
        if draw:
            cv2.putText(
                img,
                f"{int(brightness_percent)}%",
                (50, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                colorOrange,
                3,
            )
            cv2.rectangle(img,(50,120),(80,320), colorOrange, 3)
            cv2.rectangle(img,(50,320-int(brightness_bar)),(80,320), colorOrange, cv2.FILLED)

        # If the difference between the new brightness and the old one is greater than the threshold, then change the brightness
        if abs(currentBrightness - new_brightness) >= threshold:
            screen_brightness_control.set_brightness(new_brightness)
            print("Brightness Changed")
            
    def altTabChange(self, img, landmarks):
        """
        Performs an Alt+Tab action based on hand gesture.

        Explanation:
        - Initiates an Alt+Tab action when the index and thumb tips make contact.
        - Simulates the Alt key press, switches windows using Tab, and releases the Alt key.

        Args:
            img: The input image containing the hand landmarks.
            landmarks: List of landmark coordinates.

        Returns:
            None. Triggers an Alt+Tab action based on hand gesture recognition.
        """
        length, img = self.findDistance(4,8,img,landmarks,draw=True)
        
        # Index and Thumb tips make contact
        if length < 40:
            pyautogui.keyDown('alt')
            time.sleep(.2)
            pyautogui.press('tab')
            time.sleep(.2)
            pyautogui.keyUp('alt')
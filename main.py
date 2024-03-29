# This Python code snippet is using the OpenCV library (`cv2`) for computer vision tasks and a custom
# `GestureReconition` module for recognizing hand gestures. Here's a breakdown of what the code is doing:
import cv2
import GestureReconition

recog = GestureReconition.GestureRecognizer()
cap = cv2.VideoCapture(0)

# For OpenCV window
window_width = 640
window_height = 480

while True:
    success, img = cap.read()
    landmarks, img = recog.getLandmarks(img=img, draw=False)
    
    img = cv2.resize(img, (window_width, window_height))
    
    if landmarks:
        handIsThere, is_left_hand = recog.isLeftHand(img=img)
        fingers, img = recog.fingerUp(img,landmarks,is_left_hand)
        
        if handIsThere == True and is_left_hand == False:
            recog.scrollControl(fingers,scrollSpeed = 100)

            recog.mouseControl(img, landmarks,fingers,draw=True)

            # Volume control works only when index, thumb, and pinky finger are up and the middle and ring fingers are down
            # Kind of like spiderman's web shooting thingy
            if fingers == [1,1,0,0,1]:
                recog.volumeControl(img, landmarks, draw = True)
        
        if is_left_hand:
            # Similar to volume control gesture, but with Left Hand
            if fingers == [1,1,0,0,1]:
                recog.brightnessControl(img,landmarks,draw=True)
            
            # Like Nice Emoji xD
            if fingers[1:] == [0,1,1,1]:
                recog.altTabChange(img, landmarks)
        cv2.imshow("Video Feed:",img)

    key = cv2.waitKey(1)
    if key == 27: # escape
        break

cap.release()
cv2.destroyAllWindows()

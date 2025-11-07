import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time

# Open the first webcam (usually 0)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Images/T"
counter = 0

imgWhite = None

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize,3),np.uint8)*255
    
        
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)  
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)  

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue  

        if imgCrop.size != 0 or imgCrop.shape[0] > 0 or imgCrop.shape[1] > 0:
            imgCropShape = imgCrop.shape 

        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite [:, wGap:wCal+wGap] = imgResize

        else: 
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite [hGap:hCal+hGap, :] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Show the frame
    cv2.imshow("Webcam Feed", img)

    key = cv2.waitKey(1) & 0xFF

    # Exit on 'q' key
    if key == ord('q'):
        break

    if key == ord("s"):
        if imgWhite is not None:
            counter += 1
            cv2.imwrite(f'{folder}/Images_{time.time()}.jpg', imgWhite)
            print(f"Saved image {counter}")
        else:
            print("no hand")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

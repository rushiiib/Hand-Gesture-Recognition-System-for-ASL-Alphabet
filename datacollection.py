import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time
import os

# Try different camera indices
# 0 = default camera, 1 = external/continuity camera
cap = cv2.VideoCapture(0)

# Set camera properties explicitly
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Add a small delay to let camera initialize
time.sleep(1)

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Images/N"
counter = 0

# Create folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

imgWhite = None

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        exit()

print("Camera opened successfully! Press 's' to save images, 'q' to quit.")

while True:
    # Read frame
    success, img = cap.read()
    
    if not success or img is None:
        print("Failed to grab frame, retrying...")
        time.sleep(0.1)
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Calculate crop boundaries with safety checks
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)  
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)  

        imgCrop = img[y1:y2, x1:x2]

        # Check if crop is valid
        if imgCrop.size == 0 or imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue  

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                if wCal > 0:
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
            else: 
                k = imgSize / w
                hCal = math.ceil(k * h)
                if hCal > 0:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        except Exception as e:
            print(f"Error processing image: {e}")
            continue

    # Show the frame
    cv2.imshow("Webcam Feed", img)

    key = cv2.waitKey(1) & 0xFF

    # Exit on 'q' key
    if key == ord('q'):
        break

    if key == ord("s"):
        if imgWhite is not None:
            counter += 1
            filename = f'{folder}/Image_{counter}_{time.time()}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"Saved image {counter} to {filename}")
        else:
            print("No hand detected - cannot save image")

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
print(f"Total images saved: {counter}")
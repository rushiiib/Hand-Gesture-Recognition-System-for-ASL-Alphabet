import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math, time
import tensorflow as tf

# ---------------- CONFIG ----------------
MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"
IMG_SIZE = 128          # MUST match train.py
offset = 20
save_folder = "Images/T"
counter = 0

# ------------- LOAD MODEL + LABELS -------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABELS_PATH, "r") as f:
    labels = [ln.strip() for ln in f if ln.strip()]
print("Labels:", labels)

# ------------- SETUP WEBCAM + HAND DETECTOR -------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=1, detectionCon=0.8)

print("Press 'q' to quit, 's' to save the current cropped image.")

while True:
    ok, img = cap.read()
    if not ok:
        break

    imgOutput = img.copy()

    # Let cvzone draw skeleton on img (same as training)
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # clamp crop bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        if y2 > y1 and x2 > x1:
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                # create white canvas
                imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                crop_h, crop_w = imgCrop.shape[:2]
                aspectRatio = crop_h / max(1, crop_w)

                if aspectRatio > 1:   # tall
                    k = IMG_SIZE / crop_h
                    wCal = math.ceil(k * crop_w)
                    imgResize = cv2.resize(imgCrop, (wCal, IMG_SIZE))
                    wGap = math.ceil((IMG_SIZE - wCal) / 2)
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:                 # wide
                    k = IMG_SIZE / crop_w
                    hCal = math.ceil(k * crop_h)
                    imgResize = cv2.resize(imgCrop, (IMG_SIZE, hCal))
                    hGap = math.ceil((IMG_SIZE - hCal) / 2)
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # ---------- PREDICTION ----------
                img_input = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
                img_input = img_input.astype("float32") / 255.0
                img_input = np.expand_dims(img_input, axis=0)

                preds = model.predict(img_input, verbose=0)[0]
                idx = int(np.argmax(preds))
                conf = float(preds[idx])
                letter = labels[idx] if 0 <= idx < len(labels) else "?"

                # print top-3 for debugging
                top3_idx = np.argsort(preds)[-3:][::-1]
                top3_text = ", ".join(
                    [f"{labels[i]}:{preds[i]:.2f}" for i in top3_idx]
                )
                print(f"Prediction: {letter} ({conf:.2f}) | Top-3: {top3_text}")

                # ---------- DRAW PREDICTION ----------
                cv2.rectangle(imgOutput,
                              (x, y - 50),
                              (x + 260, y),
                              (255, 0, 255),
                              cv2.FILLED)
                cv2.putText(imgOutput,
                            f"{letter} {conf:.2f}",
                            (x + 10, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 255),
                            2)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Webcam Feed", imgOutput)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('s') and 'imgWhite' in locals():
        os.makedirs(save_folder, exist_ok=True)
        counter += 1
        filename = f"{save_folder}/Image_{counter}_{time.time()}.jpg"
        cv2.imwrite(filename, imgWhite)
        print(f"Saved image {counter} -> {filename}")

cap.release()
cv2.destroyAllWindows()

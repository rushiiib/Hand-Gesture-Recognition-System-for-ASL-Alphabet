# predict.py
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ------------------------------
# PARAMETERS
# ------------------------------
IMG_SIZE = 64  # same as training
MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"
TEST_DIR = "Test"  # folder with images to predict

# ------------------------------
# LOAD MODEL AND LABELS
# ------------------------------
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ------------------------------
# PREDICT FUNCTION
# ------------------------------
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    return labels[class_idx], pred[0][class_idx]

# ------------------------------
# PREDICT ON ALL IMAGES IN TEST_DIR
# ------------------------------
if not os.path.exists(TEST_DIR):
    print(f"Test folder '{TEST_DIR}' not found!")
else:
    for img_name in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, img_name)
        label, confidence = predict_image(img_path)
        print(f"{img_name} --> {label} (confidence: {confidence:.2f})")

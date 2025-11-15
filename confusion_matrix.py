import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------
IMG_SIZE = 128   # or 300, whatever your model was trained on
BASE_DIR = "/Users/aarushibhatnagar/Desktop/School/fall 2025/cmpt 310/Project/split"
MODEL_PATH = "converted_keras/keras_model.h5"
LABELS_PATH = "converted_keras/labels.txt"

# ------------------------------
# LOAD LABELS
# ------------------------------
with open(LABELS_PATH, "r") as f:
    labels = [ln.strip() for ln in f if ln.strip()]

# ------------------------------
# LOAD VALIDATION DATA
# ------------------------------
def load_data(split):
    X, y = [], []
    split_dir = os.path.join(BASE_DIR, split)
    class_names = sorted(os.listdir(split_dir))

    for idx, cname in enumerate(class_names):
        folder = os.path.join(split_dir, cname)
        if not os.path.isdir(folder):
            continue
        for img_name in os.listdir(folder):
            if img_name.startswith("."):
                continue
            img = cv2.imread(os.path.join(folder, img_name))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)
    return X, y

print("Loading validation data...")
X_val, y_val = load_data("val")

# ------------------------------
# LOAD MODEL
# ------------------------------
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

# ------------------------------
# PREDICT
# ------------------------------
print("Predicting...")
y_pred = np.argmax(model.predict(X_val), axis=1)

# ------------------------------
# CONFUSION MATRIX
# ------------------------------
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(14, 14))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("ASL Confusion Matrix (Validation Set)")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

print("✅ Confusion matrix saved as confusion_matrix.png")

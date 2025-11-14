# train.py (updated for split dataset)
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ------------------------------
# PARAMETERS
# ------------------------------
IMG_SIZE = 64
BASE_DIR = r"C:\Users\riyaa\OneDrive\Desktop\CMPT310 Project\Split"  # must contain train/val/test folders
EPOCHS = 15
BATCH_SIZE = 16

# ------------------------------
# HELPER FUNCTION TO LOAD DATA
# ------------------------------
def load_data(split):
    X = []
    y = []
    split_dir = os.path.join(BASE_DIR, split)
    labels = sorted(os.listdir(split_dir))
    
    for idx, label in enumerate(labels):
        folder_path = os.path.join(split_dir, label)
        if not os.path.exists(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)
    X = np.array(X, dtype='float32') / 255.0
    y = to_categorical(y, num_classes=len(labels))
    return X, y, labels

# ------------------------------
# LOAD TRAINING AND VALIDATION DATA
# ------------------------------
print("Loading training data...")
X_train, y_train, labels = load_data('train')

print("Loading validation data...")
X_val, y_val, _ = load_data('val')

print(f"Total training images: {len(X_train)}, Total validation images: {len(X_val)}, Total classes: {len(labels)}")

# ------------------------------
# BUILD CNN MODEL
# ------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# TRAIN MODEL
# ------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ------------------------------
# SAVE MODEL AND LABELS
# ------------------------------
os.makedirs("converted_keras", exist_ok=True)
model.save("converted_keras/keras_model.h5")

with open("converted_keras/labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

# ------------------------------
# PLOT ACCURACY AND LOSS
# ------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.savefig("training_plots.png")
plt.show()

print("Training complete. Model and labels saved!")

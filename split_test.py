import os
import shutil
import random

source = "Images"
destination = "Data"
train_ratio = 0.8

classes = ["U", "V", "W", "X", "Y", "Z"]

for c in classes:
    src_path = os.path.join(source, c)
    train_path = os.path.join(destination, "Train", c)
    test_path = os.path.join(destination, "Test", c)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    files = os.listdir(src_path)
    random.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for f in train_files:
        shutil.copy(os.path.join(src_path, f), train_path)

    for f in test_files:
        shutil.copy(os.path.join(src_path, f), test_path)

print("Dataset split complete!")

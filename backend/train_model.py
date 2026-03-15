import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = r"C:\Users\LVRSS\Assistive-Communication-System\backend\data\data"

print("Loading dataset...")

if not os.path.exists(DATASET_PATH):
    print("❌ Dataset path not found!")
    exit()

data = []
labels = []

folders = os.listdir(DATASET_PATH)
print("Found folders:", folders)

for folder in folders:
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    print("Reading folder:", folder)

    files = os.listdir(folder_path)
    print("Number of files:", len(files))

    for file in files:
        file_path = os.path.join(folder_path, file)

        img = cv2.imread(file_path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.flatten()

        data.append(img)
        labels.append(folder)

print("Total samples loaded:", len(data))

if len(data) == 0:
    print("❌ No images found.")
    exit()

data = np.array(data)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

print("Training model...")

model = RandomForestClassifier(n_estimators=100)
model.fit(data, labels_encoded)

with open("model.p", "wb") as f:
    pickle.dump({"model": model, "label_encoder": le}, f)

print("✅ Training complete.")
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Dataset path
DATASET_PATH = "dataset/indian"
MODEL_SAVE_PATH = "models/isl_model.pkl"
LABEL_SAVE_PATH = "models/label_encoder.pkl"

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

X = []
y = []

print("🔄 Processing dataset...")

for label_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label_folder)

    if not os.path.isdir(folder_path):
        continue

    for image_file in tqdm(os.listdir(folder_path), desc=f"Processing {label_folder}"):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                X.append(landmarks)
                y.append(label_folder)   # 🔥 IMPORTANT: use folder name

hands.close()

X = np.array(X)
y = np.array(y)

print("📊 Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Training model...")
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y_encoded)

# Save model
os.makedirs("models", exist_ok=True)

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)

with open(LABEL_SAVE_PATH, "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained and saved successfully!")
print("Classes:", le.classes_)
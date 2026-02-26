import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATASET_PATH = "dataset/images_for_phrases"
MODEL_SAVE_PATH = "models/phrases_model.pkl"
LABEL_SAVE_PATH = "models/phrases_labels.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

X = []
y = []

print("🔄 Processing phrase dataset...")

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

            all_landmarks = []

            # Combine both hands
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    all_landmarks.extend([lm.x, lm.y])

            # Pad if only 1 hand detected
            if len(results.multi_hand_landmarks) == 1:
                all_landmarks.extend([0] * 42)

            if len(all_landmarks) == 84:
                X.append(all_landmarks)
                y.append(label_folder)

hands.close()

X = np.array(X)
y = np.array(y)

print("📊 Encoding phrase labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Training phrase model...")
model = RandomForestClassifier(n_estimators=300)
model.fit(X, y_encoded)

os.makedirs("models", exist_ok=True)

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)

with open(LABEL_SAVE_PATH, "wb") as f:
    pickle.dump(le, f)

print("✅ Phrase model trained successfully!")
print("Phrase Classes:", le.classes_)
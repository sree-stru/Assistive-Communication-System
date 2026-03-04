import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----------------------------
# Paths
# ----------------------------
DATASET_PATH = "dataset/indian"
MODEL_SAVE_PATH = "models/isl_model.pkl"
LABEL_SAVE_PATH = "models/label_encoder.pkl"

# ----------------------------
# MediaPipe Setup (2 Hands)
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2
)

X = []
y = []

print("🔄 Processing dataset...")

# ----------------------------
# Dataset Processing
# ----------------------------
for label_folder in sorted(os.listdir(DATASET_PATH)):
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

        if not results.multi_hand_landmarks:
            continue

        landmarks_all = []

        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = []

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x - base_x)
                landmarks.append(lm.y - base_y)

            landmarks = np.array(landmarks)

            # Scale normalization
            max_value = np.max(np.abs(landmarks))
            if max_value != 0:
                landmarks = landmarks / max_value

            landmarks_all.extend(landmarks.tolist())

        # If only 1 hand detected → pad second hand
        if len(landmarks_all) == 42:
            landmarks_all.extend([0] * 42)

        # Ensure exactly 84 features
        landmarks_all = landmarks_all[:84]

        X.append(landmarks_all)
        y.append(label_folder)

hands.close()

X = np.array(X)
y = np.array(y)

print("📊 Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ----------------------------
# Model Training
# ----------------------------
print("Training model...")
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_test, y_test)

print("Training Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)

print("\nDetailed Validation Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ----------------------------
# Save Model
# ----------------------------
os.makedirs("models", exist_ok=True)

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)

with open(LABEL_SAVE_PATH, "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained and saved successfully!")
print("Classes:", le.classes_)
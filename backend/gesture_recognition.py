"""
Stable Gesture Recognition Module
Single-hand model with:
- Proper normalization (x, y, z)
- Confidence threshold
- Stability filtering
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os


class GestureRecognizer:
    def __init__(self):

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Model
        self.model = None
        self.label_encoder = None

        # Stability
        self.last_predictions = []
        self.stable_prediction = None

        # Confidence threshold
        self.conf_threshold = 0.75

        self.load_model()

    # -------------------------------
    # Load Model
    # -------------------------------
    def load_model(self):
        try:
            if os.path.exists("models/isl_model.pkl") and os.path.exists("models/label_encoder.pkl"):
                with open("models/isl_model.pkl", "rb") as f:
                    self.model = pickle.load(f)

                with open("models/label_encoder.pkl", "rb") as f:
                    self.label_encoder = pickle.load(f)

                print("✅ Model loaded successfully")
            else:
                print("❌ Model files not found inside models/ folder")
        except Exception as e:
            print("Model loading error:", e)

    # -------------------------------
    # Extract & Normalize Landmarks
    # -------------------------------
    def extract_landmarks(self, hand_landmarks):

        landmarks = []

        # Wrist reference normalization
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z

        for lm in hand_landmarks.landmark:
            landmarks.extend([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z
            ])

        return landmarks  # 63 features

    # -------------------------------
    # Predict Gesture
    # -------------------------------
    def predict(self, features):

        if self.model is None:
            return "Unknown"

        try:
            features = np.array(features).reshape(1, -1)

            probs = self.model.predict_proba(features)
            confidence = np.max(probs)

            if confidence < self.conf_threshold:
                return "Unknown"

            pred = int(self.model.predict(features)[0])
            label = self.label_encoder.inverse_transform([pred])[0]

            return str(label)

        except Exception as e:
            print("Prediction error:", e)
            return "Unknown"

    # -------------------------------
    # Main Recognition Function
    # -------------------------------
    def recognize_gesture(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture_text = "No hand detected"

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            # Extract features
            features = self.extract_landmarks(hand_landmarks)

            # Predict
            prediction = self.predict(features)

            # Stability filtering
            self.last_predictions.append(prediction)

            if len(self.last_predictions) > 10:
                self.last_predictions.pop(0)

            if self.last_predictions.count(prediction) > 7:
                self.stable_prediction = prediction

            gesture_text = self.stable_prediction if self.stable_prediction else "Detecting..."

        else:
            self.stable_prediction = None
            self.last_predictions.clear()

        return frame, gesture_text

    def cleanup(self):
        self.hands.close()
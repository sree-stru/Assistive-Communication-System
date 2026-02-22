"""
Advanced Gesture Recognition Module
Supports:
- Indian Alphabets & Digits Model (1-hand)
- Phrases Model (1-hand & 2-hand mixed)
- Confidence-based prediction
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os


class GestureRecognizer:
    def __init__(self):
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Models
        self.indian_model = None
        self.indian_labels = None
        self.phrases_model = None
        self.phrases_labels = None

        # Confidence threshold
        self.phrase_conf_threshold = 0.60

        self.load_models()

    # ---------------------------------------------------
    # LOAD MODELS
    # ---------------------------------------------------
    def load_models(self):
        try:
            # Indian model
            if os.path.exists("models/isl_model.pkl") and os.path.exists("models/label_encoder.pkl"):
                with open("models/isl_model.pkl", "rb") as f:
                    self.indian_model = pickle.load(f)
                with open("models/label_encoder.pkl", "rb") as f:
                    self.indian_labels = pickle.load(f)
                print("✅ Indian model loaded")
            else:
                print("⚠ Indian model files not found")

            # Phrase model
            if os.path.exists("models/phrases_model.pkl") and os.path.exists("models/phrases_labels.pkl"):
                with open("models/phrases_model.pkl", "rb") as f:
                    self.phrases_model = pickle.load(f)
                with open("models/phrases_labels.pkl", "rb") as f:
                    self.phrases_labels = pickle.load(f)
                print("✅ Phrases model loaded")
            else:
                print("⚠ Phrase model files not found")

        except Exception as e:
            print("❌ Model loading error:", e)

    # ---------------------------------------------------
    # EXTRACT SINGLE HAND LANDMARKS (42 features)
    # ---------------------------------------------------
    def extract_single_hand(self, hand_landmarks):
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
        return landmarks

    # ---------------------------------------------------
    # PREDICTION LOGIC
    # ---------------------------------------------------
    def predict_with_models(self, combined_landmarks):
        features = np.array(combined_landmarks).reshape(1, -1)

        phrase_label = None
        phrase_conf = 0

        # ----------------------------
        # Try Phrase Model (84 features)
        # ----------------------------
        if self.phrases_model is not None and self.phrases_labels is not None:
            try:
                probs = self.phrases_model.predict_proba(features)
                phrase_conf = float(np.max(probs))
                pred = int(self.phrases_model.predict(features)[0])
                phrase_label = self.phrases_labels.inverse_transform([pred])[0]
            except Exception as e:
                print("Phrase model error:", e)

        # If confident → return phrase
        if phrase_label and phrase_conf >= self.phrase_conf_threshold:
            print("PHRASE MODEL OUTPUT:", phrase_label, "Confidence:", phrase_conf)
            return str(phrase_label)

        # ----------------------------
        # Fallback to Indian Model (first 42 features only)
        # ----------------------------
        if self.indian_model is not None and self.indian_labels is not None:
            try:
                single_hand_features = np.array(combined_landmarks[:42]).reshape(1, -1)
                pred = int(self.indian_model.predict(single_hand_features)[0])
                label = self.indian_labels.inverse_transform([pred])[0]
                print("INDIAN MODEL OUTPUT:", label)
                return str(label)
            except Exception as e:
                print("Indian model error:", e)

        return "Unknown"

    # ---------------------------------------------------
    # MAIN FRAME PROCESSING
    # ---------------------------------------------------
    def recognize_gesture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_text = "No hand detected"

        if results.multi_hand_landmarks:

            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

            combined_landmarks = []

            # Extract detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                combined_landmarks.extend(
                    self.extract_single_hand(hand_landmarks)
                )

            # 🔥 Always make feature size 84
            if len(combined_landmarks) == 42:
                combined_landmarks.extend([0] * 42)

            # If more than 2 hands somehow detected, trim
            combined_landmarks = combined_landmarks[:84]

            gesture_text = self.predict_with_models(combined_landmarks)

        return frame, gesture_text

    # ---------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------
    def cleanup(self):
        self.hands.close()
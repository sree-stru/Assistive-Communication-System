"""
Advanced Gesture Recognition Module
Supports:
- Indian Alphabets & Digits Model (2-hand, 84 features)
- Proper scale normalization
- Stable prediction pipeline
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os


class GestureRecognizer:
    def __init__(self):
        # ----------------------------
        # MediaPipe Hands
        # ----------------------------
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # ----------------------------
        # Models
        # ----------------------------
        self.indian_model = None
        self.indian_labels = None

        self.load_models()

    # ---------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------
    def load_models(self):
        try:
            if os.path.exists("models/isl_model.pkl") and os.path.exists("models/label_encoder.pkl"):
                with open("models/isl_model.pkl", "rb") as f:
                    self.indian_model = pickle.load(f)

                with open("models/label_encoder.pkl", "rb") as f:
                    self.indian_labels = pickle.load(f)

                print("✅ Indian model loaded (84 features)")
            else:
                print("⚠ Model files not found.")

        except Exception as e:
            print("❌ Model loading error:", e)

    # ---------------------------------------------------
    # NORMALIZE HAND LANDMARKS (Same as Training)
    # ---------------------------------------------------
    def normalize_landmarks(self, hand_landmarks):
        landmarks = []

        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y

        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x - base_x)
            landmarks.append(lm.y - base_y)

        landmarks = np.array(landmarks)

        max_value = np.max(np.abs(landmarks))
        if max_value != 0:
            landmarks = landmarks / max_value

        return landmarks.tolist()

    # ---------------------------------------------------
    # MAIN FRAME PROCESSING
    # ---------------------------------------------------
    def recognize_gesture(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gesture_text = "No hand detected"

        if results.multi_hand_landmarks:

            combined_landmarks = []

            # Sort hands left-to-right
            sorted_hands = sorted(
               results.multi_hand_landmarks,
               key=lambda hand: hand.landmark[0].x
       )

            for hand_landmarks in sorted_hands:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Normalize and extract features
                normalized = self.normalize_landmarks(hand_landmarks)
                combined_landmarks.extend(normalized)

            # If only 1 hand detected → pad second hand
            if len(combined_landmarks) == 42:
                combined_landmarks.extend([0] * 42)

            # Ensure exactly 84 features
            combined_landmarks = combined_landmarks[:84]

            # Convert to numpy
            features = np.array(combined_landmarks).reshape(1, -1)

            # Predict
            if self.indian_model is not None:
                try:
                    pred = int(self.indian_model.predict(features)[0])
                    label = self.indian_labels.inverse_transform([pred])[0]

                    gesture_text = str(label)

                except Exception as e:
                    print("Prediction error:", e)
                    gesture_text = "Prediction Error"

        return frame, gesture_text

    # ---------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------
    def cleanup(self):
        self.hands.close()
"""
Gesture Recognition Module
Handles hand detection, landmark extraction, and gesture classification
"""
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load trained model if exists
        self.model = None
        self.label_encoder = None
        self.load_model()
        
        # Simple gesture mapping for demo (can be replaced with ML model)
        self.gesture_map = {
            'thumbs_up': 'Hello',
            'peace': 'Thank you',
            'fist': 'Yes',
            'open_palm': 'Help',
            'ok_sign': 'Okay'
        }
    
    def load_model(self):
        """Load the trained gesture recognition model"""
        model_path = 'models/gesture_model.pkl'
        encoder_path = 'models/label_encoder.pkl'
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks"""
        landmarks = []
        
        # Get all 21 landmarks (x, y, z coordinates)
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position"""
        landmarks = landmarks.reshape(-1, 3)
        wrist = landmarks[0]
        
        # Subtract wrist position to make relative
        normalized = landmarks - wrist
        
        # Flatten back
        return normalized.flatten()
    
    def detect_simple_gesture(self, hand_landmarks):
        """Simple rule-based gesture detection for demo purposes"""
        # Extract specific landmark positions
        landmarks = hand_landmarks.landmark
        
        # Thumb tip and thumb IP
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        
        # Index finger tip and PIP
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        
        # Middle finger tip and PIP
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        
        # Ring finger tip and PIP
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        
        # Pinky tip and PIP
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Count extended fingers
        fingers_up = 0
        
        # Thumb (check if tip is to the right/left of IP)
        if thumb_tip.x < thumb_ip.x - 0.05:
            fingers_up += 1
        
        # Other fingers (check if tip is above PIP)
        if index_tip.y < index_pip.y:
            fingers_up += 1
        if middle_tip.y < middle_pip.y:
            fingers_up += 1
        if ring_tip.y < ring_pip.y:
            fingers_up += 1
        if pinky_tip.y < pinky_pip.y:
            fingers_up += 1
        
        # Map finger count to gestures
        if fingers_up == 0:
            return "Fist - YES"
        elif fingers_up == 1:
            return "One - I"
        elif fingers_up == 2:
            return "Two - NEED"
        elif fingers_up == 3:
            return "Three - HELP"
        elif fingers_up == 4:
            return "Four - WAIT"
        elif fingers_up == 5:
            return "Open Palm - HELLO"
        else:
            return "Unknown"
    
    def recognize_gesture(self, frame):
        """
        Main method to recognize gesture from a frame
        Returns: processed_frame, gesture_text
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        gesture_text = "No hand detected"
        
        # If hand detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Recognize gesture
                if self.model is not None:
                    # Use ML model if available
                    landmarks = self.extract_landmarks(hand_landmarks)
                    normalized = self.normalize_landmarks(landmarks)
                    prediction = self.model.predict([normalized])[0]
                    gesture_text = self.label_encoder.inverse_transform([prediction])[0]
                else:
                    # Use simple rule-based detection
                    gesture_text = self.detect_simple_gesture(hand_landmarks)
        
        return frame, gesture_text
    
    def cleanup(self):
        """Release resources"""
        self.hands.close()

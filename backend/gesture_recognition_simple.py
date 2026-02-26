"""
Simple Gesture Recognition Module
Uses OpenCV and basic hand detection
"""
import cv2
import numpy as np


class GestureRecognizer:
    """
    Simple gesture recognizer without MediaPipe (fallback)
    Uses basic contour detection
    """
    
    def __init__(self):
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    def detect_hand(self, frame):
        """Simple hand detection using HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        return mask
    
    def count_fingers(self, mask, frame):
        """Count extended fingers"""
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return -1
        
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt, returnPoints=True)
        
        return len(hull)
    
    def recognize_gesture(self, frame):
        """Recognize gesture from frame"""
        mask = self.detect_hand(frame)
        finger_count = self.count_fingers(mask, frame)
        
        if finger_count == -1:
            return "No hand detected"
        elif finger_count <= 5:
            return f"YES"
        else:
            return f"HELLO"
    
    def process_frame(self, frame):
        """Process frame and return result"""
        gesture = self.recognize_gesture(frame)
        return frame, gesture


# For testing with MediaPipe-like API
try:
    import mediapipe as mp
    
    class GestureRecognizer:
        """Main Gesture Recognizer with MediaPipe"""
        
        def __init__(self):
            try:
                # Try new MediaPipe API
                self.mp_hands = mp.tasks.vision.HandLandmarker
                self.use_old_api = False
            except:
                # Fallback to old API
                self.use_old_api = True
                from mediapipe.python.solutions import hands, drawing_utils
                self.hands = hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.7
                )
                self.drawing_utils = drawing_utils
        
        def process_frame(self, frame):
            """Process frame - compatible with both APIs"""
            gesture_text = "No hand detected"
            
            try:
                if self.use_old_api:
                    # Use old MediaPipe API
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        self.drawing_utils.draw_landmarks(
                            frame, hand_landmarks,
                            self.hands.HAND_CONNECTIONS
                        )
                        gesture_text = self._recognize_gesture(hand_landmarks)
            except Exception as e:
                print(f"Error: {e}")
                gesture_text = "Detection Error"
            
            return frame, gesture_text
        
        def _recognize_gesture(self, hand_landmarks):
            """Recognize gesture from landmarks"""
            landmarks = hand_landmarks.landmark
            
            # Count extended fingers
            fingers_up = 0
            
            # Thumb
            if landmarks[4].x < landmarks[3].x:
                fingers_up += 1
            
            # Other fingers
            if landmarks[8].y < landmarks[6].y:
                fingers_up += 1
            if landmarks[12].y < landmarks[10].y:
                fingers_up += 1
            if landmarks[16].y < landmarks[14].y:
                fingers_up += 1
            if landmarks[20].y < landmarks[18].y:
                fingers_up += 1
            
            if fingers_up == 0:
                return "NO"
            elif fingers_up == 1:
                return "YES"
            elif fingers_up == 2:
                return "THANK YOU"
            elif fingers_up == 5:
                return "HELLO"
            else:
                return f"Gesture_{fingers_up}"
        
        def cleanup(self):
            """Cleanup resources"""
            pass

except ImportError:
    pass

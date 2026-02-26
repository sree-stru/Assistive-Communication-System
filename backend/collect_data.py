"""
Data Collection Script (Optional - For Advanced Users)
This script helps collect training data for custom gestures
You can use this to train your own gesture recognition model
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.dataset_path = 'dataset'
        os.makedirs(self.dataset_path, exist_ok=True)
        
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as features"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist"""
        landmarks = landmarks.reshape(-1, 3)
        wrist = landmarks[0]
        normalized = landmarks - wrist
        return normalized.flatten()
    
    def collect_samples(self, gesture_name, num_samples=100):
        """Collect samples for a specific gesture"""
        print(f"\nCollecting {num_samples} samples for gesture: {gesture_name}")
        print("Press 's' to save a sample, 'q' to finish early")
        
        cap = cv2.VideoCapture(0)
        samples = []
        count = 0
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display instructions
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{num_samples}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(results.multi_hand_landmarks[0])
                    normalized = self.normalize_landmarks(landmarks)
                    samples.append(normalized)
                    count += 1
                    print(f"Sample {count} saved!")
                else:
                    print("No hand detected! Try again.")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save samples
        if samples:
            filepath = os.path.join(self.dataset_path, f"{gesture_name}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(samples, f)
            print(f"\n✓ Saved {len(samples)} samples for {gesture_name}")
        
        return samples
    
    def train_model(self):
        """Train a model using collected data"""
        print("\n=== Training Model ===")
        
        # Load all gesture data
        X = []
        y = []
        
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.pkl'):
                gesture_name = filename.replace('.pkl', '')
                filepath = os.path.join(self.dataset_path, filename)
                
                with open(filepath, 'rb') as f:
                    samples = pickle.load(f)
                
                X.extend(samples)
                y.extend([gesture_name] * len(samples))
                print(f"Loaded {len(samples)} samples for {gesture_name}")
        
        if len(X) == 0:
            print("No training data found!")
            return
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Train model
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = model.score(X_test, y_test)
        print(f"\n✓ Model trained successfully!")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        with open('models/gesture_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print("\n✓ Model saved to 'models/' directory")

def main():
    """Main function for data collection"""
    collector = DataCollector()
    
    print("=" * 50)
    print("Gesture Data Collection Tool")
    print("=" * 50)
    print("\nOptions:")
    print("1. Collect gesture samples")
    print("2. Train model from collected data")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            gesture_name = input("Enter gesture name (e.g., 'hello', 'thanks'): ").strip()
            num_samples = int(input("Number of samples to collect (default 100): ") or "100")
            collector.collect_samples(gesture_name, num_samples)
        
        elif choice == '2':
            collector.train_model()
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()

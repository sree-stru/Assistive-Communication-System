import os
import cv2
import numpy as np
import mediapipe as mp


def load_dataset(dataset_path='data/data'):
    """
    Load images from dataset/Indian/ folder and extract hand landmarks.
    Each subfolder name represents a label (A, B, 1, 2, etc.)
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        tuple: (X, y, label_names) where:
            - X is numpy array of hand landmarks (Nx42)
            - y is numpy array of labels
            - label_names is list of label names
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    X = []  # Feature vectors (landmarks)
    y = []  # Labels
    label_names = []
    
    # Get list of subdirectories (labels)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")
    
    label_dirs = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))])
    
    if not label_dirs:
        raise ValueError(f"No subdirectories found in '{dataset_path}'")
    
    # Iterate through each label directory
    for label_idx, label_name in enumerate(label_dirs):
        label_path = os.path.join(dataset_path, label_name)
        label_names.append(label_name)
        
        # Load all images in this label's directory
        image_files = [f for f in os.listdir(label_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        print(f"Processing label '{label_name}' ({len(image_files)} images)...")
        
        for image_file in image_files:
            image_path = os.path.join(label_path, image_file)
            
            try:
                # Read image
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect hands and landmarks
                results = hands.process(image_rgb)
                
                # Check if hand was detected
                if results.multi_hand_landmarks is None or len(results.multi_hand_landmarks) == 0:
                    continue
                
                # Extract landmarks from first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract 21 landmarks (x, y coordinates) as feature vector
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append(landmark.x)
                    landmarks_list.append(landmark.y)
                
                # Convert to numpy array (42 values)
                feature_vector = np.array(landmarks_list, dtype=np.float32)
                
                X.append(feature_vector)
                y.append(label_idx)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    # Convert to numpy arrays
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total images with detected hands: {len(X_array)}")
    print(f"Label names: {label_names}")
    print(f"Feature vector shape: {X_array.shape}")
    
    hands.close()
    
    return X_array, y_array, label_names


if __name__ == "__main__":
    # Example usage
    try:
        X, y, label_names = load_dataset()
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label mapping: {dict(enumerate(label_names))}")
    except Exception as e:
        print(f"Error: {e}")

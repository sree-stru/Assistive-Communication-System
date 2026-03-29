import numpy as np
import os
import json
import config

class SignLandmarkData:
    """
    Loads the trained landmark data and provides median landmark skeletons 
    for each sign language letter class.
    """
    def __init__(self):
        self.label_to_landmarks = {}
        self._load_data()

    def _load_data(self):
        data_path = os.path.join(config.BASE_DIR, config.LANDMARKS_DATA_FILE)
        labels_path = os.path.join(config.BASE_DIR, config.LANDMARKS_LABELS_FILE)
        
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            print(f" [SignLandmarkData] WARNING: Missing {data_path} or {labels_path}")
            return

        print(f" [SignLandmarkData] Loading landmark dataset for letter animation...")
        X = np.load(data_path)
        y = np.load(labels_path)
        
        with open(config.LABEL_MAP_PATH) as f:
            label_map = json.load(f)
            
        # Reverse label map: index string -> letter
        idx_to_class = {int(v): k for k, v in label_map.items()}
        
        # Group landmarks by class
        class_data = {label: [] for label in idx_to_class.values()}
        
        for i, label_idx in enumerate(y):
            if label_idx in idx_to_class:
                class_label = idx_to_class[label_idx]
                class_data[class_label].append(X[i])
                
        # Calculate median landmark for each class to get a representative skeleton
        for label, items in class_data.items():
            if items:
                # Median helps reduce outlier noise in the dataset
                median_vector = np.median(items, axis=0)
                # Reshape (42,) -> (21, 2)
                landmarks_2d = median_vector.reshape(21, 2).tolist()
                self.label_to_landmarks[label] = landmarks_2d
                
        print(f" [SignLandmarkData] Loaded median landmarks for {len(self.label_to_landmarks)} classes.")

    def get_landmarks(self, char):
        """
        Returns the 21 (x,y) normalized coordinates for a given letter/digit,
        or None if not found in the dataset.
        """
        char = str(char).upper()
        return self.label_to_landmarks.get(char, None)

# Global singleton instance loaded once
_instance = None

def get_landmark_provider():
    global _instance
    if _instance is None:
        _instance = SignLandmarkData()
    return _instance

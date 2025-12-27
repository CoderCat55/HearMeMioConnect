import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class GestureClassifier:
    """
    Legacy Classifier for main.py compatibility.
    Used by 'tr', 'cf', 'cb' commands.
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.calibration_data = {}

    def load_calibration_data(self):
        """Loads existing calibration data from .npy files in calibration_data folder."""
        if not os.path.exists('calibration_data'):
            return
        
        print("Loading legacy calibration data...")
        for file in os.listdir('calibration_data'):
            if file.endswith('.npy'):
                try:
                    # Assuming format: gesturename_timestamp.npy
                    gesture_name = file.split('_')[0]
                    data = np.load(os.path.join('calibration_data', file))
                    self.add_calibration_sample(gesture_name, data)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    def add_calibration_sample(self, gesture_name, data):
        features = self.extract_features(data)
        if gesture_name not in self.calibration_data:
            self.calibration_data[gesture_name] = []
        self.calibration_data[gesture_name].append(features)

    def train(self):
        X = []
        y = []
        for gesture, feats_list in self.calibration_data.items():
            for feats in feats_list:
                X.append(feats)
                y.append(gesture)
        
        if not X:
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.model = SVC(probability=True)
        self.model.fit(X_scaled, y)
        return True

    def classify(self, features):
        if self.model is None:
            return "Model not trained"
        
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)

    @staticmethod
    def extract_features(data):
        # Features: Mean, Std, RMS, Min, Max for each channel (34 channels)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        rms = np.sqrt(np.mean(data**2, axis=0))
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        
        return np.concatenate([mean, std, rms, min_val, max_val])
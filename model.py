"""
trains itself based on calibration samples
reads realtime data from shared memory for a fixed time and classifies returns result
"""
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import numpy as np
import pickle
import os

class GestureClassifier:
    def __init__(self):
        self.calibration_data = {}  # gesture_name -> list of np.arrays
        self.model = None
        self.scaler = StandardScaler()
        self.gesture_labels = []  # Ordered list of gesture names
    
    @staticmethod
    def extract_features(time_series_data):
        """
        Extract statistical features from time series
        time_series_data shape: (time_steps, 34) - time series from both myos
        Returns: 1D feature vector
        """
        features = []
        
        # Extract features from each channel (34 channels total)
        for channel in range(time_series_data.shape[1]):
            channel_data = time_series_data[:, channel]
            
            # Statistical features
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.max(channel_data) - np.min(channel_data),  # Range
            ])
        
        # Total features: 34 channels * 5 features = 170 features
        return np.array(features)
    
    def add_calibration_sample(self, gesture_name, time_series_data):
        """
        Add a new calibration sample
        time_series_data shape: (time_steps, 34)
        """
        if gesture_name not in self.calibration_data:
            self.calibration_data[gesture_name] = []
        
        self.calibration_data[gesture_name].append(time_series_data)
        print(f"Added calibration sample for '{gesture_name}'. Total: {len(self.calibration_data[gesture_name])}")
    
    def train(self):
        """Train SVM on calibration data"""
        if len(self.calibration_data) < 2:
            print("ERROR: Need at least 2 gestures to train!")
            return False
        
        # Check if there's enough data per gesture
        for gesture, samples in self.calibration_data.items():
            if len(samples) < 2:
                print(f"WARNING: Gesture '{gesture}' has only {len(samples)} sample. Training may be less accurate or fail if it's the only one in the test split.")
        
        print("Extracting features from calibration data...")
        X = []  # Features
        y = []  # Labels
        
        self.gesture_labels = sorted(self.calibration_data.keys())
        label_map = {name: i for i, name in enumerate(self.gesture_labels)}
        
        for gesture_name in self.gesture_labels:
            samples = self.calibration_data[gesture_name]
            for time_series in samples:
                features = self.extract_features(time_series)
                X.append(features)
                # Use numeric labels for stratify
                y.append(label_map[gesture_name])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training on {len(X)} samples from {len(self.gesture_labels)} gestures...")
        
        # Normalize features
        # Split data to prevent data leakage during scaling and for validation
        # stratify=y ensures both train and test sets have proportional gesture representation
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            print("Could not split data for validation, likely due to too few samples for a gesture. Training on all data.")
            X_train, X_test, y_train, y_test = X, [], y, []

        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train SVM
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate and print accuracy on the test set
        accuracy = self.model.score(self.scaler.transform(X_test), y_test) if len(X_test) > 0 else 1.0
        print(f"Training complete! Validation Accuracy: {accuracy:.2f}")
        return True, accuracy
    
    def classify(self, features):
        """
        Classify a feature vector
        features: 1D numpy array of features
        Returns: gesture name (string)
        """
        if self.model is None:
            return "ERROR: Model not trained yet!"
        
        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Normalize
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction_idx = self.model.predict(features_scaled)[0]
        return self.gesture_labels[prediction_idx]
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if self.model is None:
            print("ERROR: No model to save!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'gesture_labels': self.gesture_labels,
            'calibration_data': self.calibration_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.gesture_labels = model_data['gesture_labels']
        self.calibration_data = model_data['calibration_data']
        
        print(f"Model loaded from {filepath}")
        return True
    
    def load_calibration_data(self, data_dir='calibration_data'):
        """Load previously saved calibration from disk"""
        if not os.path.isdir(data_dir):
            print("No calibration data directory found")
            return
        
        # Look for .csv files now
        files = glob.glob(os.path.join(data_dir, '*.csv'))
        if not files:
            print("No .csv calibration files found")
            return
        
        import pandas as pd
        for file in files:
            # Extract gesture name from filename (remove timestamp)
            basename = os.path.basename(file)
            gesture_name = basename.split('_')[0]  # Get part before first underscore
            
            data = pd.read_csv(file, header=None).values
            if gesture_name not in self.calibration_data:
                self.calibration_data[gesture_name] = []
            self.calibration_data[gesture_name].append(data)
        
        print(f"Loaded calibration data for {len(self.calibration_data)} gestures")
"""
trains itself based on calibration samples
reads realtime data from shared memory for a fixed time and classifies returns result
"""
from sklearn import svm
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
        hangi featureların çıkarılacağını buraya yazıyoruz 
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
        
        print("Extracting features from calibration data...")
        X = []  # Features
        y = []  # Labels
        
        self.gesture_labels = sorted(self.calibration_data.keys())
        
        for gesture_name in self.gesture_labels:
            samples = self.calibration_data[gesture_name]
            for time_series in samples:
                features = self.extract_features(time_series)
                X.append(features)
                y.append(gesture_name)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training on {len(X)} samples from {len(self.gesture_labels)} gestures...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        self.model.fit(X_scaled, y)
        
        print("Training complete!")
        return True
    
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
        prediction = self.model.predict(features_scaled)
        return prediction[0]
    
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
    
    def load_calibration_data(self):
        """Load previously saved calibration from disk"""
        if not os.path.exists('calibration_data'):
            print("No calibration data directory found")
            return
        
        files = glob.glob('calibration_data/*.npy')
        if not files:
            print("No calibration files found")
            return
        
        for file in files:
            # Extract gesture name from filename (remove timestamp)
            basename = os.path.basename(file)
            gesture_name = basename.split('_')[0]  # Get part before first underscore
            
            data = np.load(file)
            if gesture_name not in self.calibration_data:
                self.calibration_data[gesture_name] = []
            self.calibration_data[gesture_name].append(data)
        
        print(f"Loaded calibration data for {len(self.calibration_data)} gestures")
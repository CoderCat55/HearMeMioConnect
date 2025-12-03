"""
trains itself based on calibration samples
reads realtime data from shared memory for a fixed time and classifies returns result
"""
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import glob
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

class GestureClassifier:
    def __init__(self):
        self.calibration_data = {}  # gesture_name -> list of np.arrays
        self.model = None
        self.scaler = StandardScaler()
        self.gesture_labels = []  # Ordered list of gesture names
    
    @staticmethod
    def extract_features(time_series_data):
        """
        Extract enhanced statistical features from time series
        time_series_data shape: (time_steps, 34) - time series from both myos
        Returns: 1D feature vector
        """
        features = []
        
        for channel in range(time_series_data.shape[1]):
            channel_data = time_series_data[:, channel]
            
            # Enhanced statistical features
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.max(channel_data) - np.min(channel_data),  # Range
                np.median(channel_data),
                skew(channel_data),
                kurtosis(channel_data),
                np.sqrt(np.mean(channel_data**2)),  # RMS
            ])
        
        # Total features: 34 channels * 9 features = 306 features
        return np.array(features)
    
    def add_calibration_sample(self, gesture_name, time_series_data):
        """Add a new calibration sample"""
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
        """Classify a feature vector"""
        if self.model is None:
            return "ERROR: Model not trained yet!"
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
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
            basename = os.path.basename(file)
            gesture_name = basename.split('_')[0]
            data = np.load(file)
            if gesture_name not in self.calibration_data:
                self.calibration_data[gesture_name] = []
            self.calibration_data[gesture_name].append(data)
        print(f"Loaded calibration data for {len(self.calibration_data)} gestures")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on a test set and display confusion matrix
        X_test: list of np.arrays (time series)
        y_test: list of labels
        """
        if self.model is None:
            print("ERROR: Model not trained!")
            return
        # Extract features
        X_features = np.array([self.extract_features(x) for x in X_test])
        # Predict
        y_pred = [self.classify(x) for x in X_features]
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.gesture_labels)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        # Plot confusion matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.gesture_labels,
                    yticklabels=self.gesture_labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

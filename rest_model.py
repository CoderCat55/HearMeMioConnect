# rest_model.py
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

class RestModel:
    def __init__(self, window_size_ms=20, sampling_rate=200):
        self.model = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        self.scaler = StandardScaler()
        self.window_size_ms = window_size_ms
        self.sampling_rate = sampling_rate
        
        # Calculate samples per window
        self.samples_per_window = int((window_size_ms / 1000) * sampling_rate)  # 4 samples
        self.stride = self.samples_per_window // 2  # 50% overlap = 2 samples stride
        
    @staticmethod
    def extract_features(time_series_data):
        """
        Same feature extraction as GestureClassifier
        time_series_data shape: (time_steps, 34)
        Returns: 1D feature vector (170 features)
        """
        features = []
        for channel in range(time_series_data.shape[1]):
            channel_data = time_series_data[:, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.max(channel_data) - np.min(channel_data),  # Range
            ])
        return np.array(features)
    
    def train(self, rest_data_list):
        """
        Train on all participants' rest data
        rest_data_list: list of numpy arrays, each shape (time_steps, 34)
        """
        print("Extracting features from rest data with sliding windows...")
        X = []
        
        for rest_sample in rest_data_list:
            # Slide window with 50% overlap
            num_windows = (len(rest_sample) - self.samples_per_window) // self.stride + 1
            
            for i in range(num_windows):
                start_idx = i * self.stride
                end_idx = start_idx + self.samples_per_window
                window = rest_sample[start_idx:end_idx]
                
                features = self.extract_features(window)
                X.append(features)
        
        X = np.array(X)
        print(f"Training RestModel on {len(X)} windows from rest data...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train One-Class SVM
        self.model.fit(X_scaled)
        print("RestModel training complete!")
        return True
    
    def is_rest(self, features):
        """
        Predict if features represent rest position
        features: 1D numpy array (170 features)
        Returns: True if rest, False if not rest
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        
        # One-Class SVM returns 1 for inliers (rest), -1 for outliers (not rest)
        return prediction[0] == 1
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'window_size_ms': self.window_size_ms,
            'sampling_rate': self.sampling_rate,
            'samples_per_window': self.samples_per_window,
            'stride': self.stride
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"RestModel saved to {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.window_size_ms = model_data['window_size_ms']
        self.sampling_rate = model_data['sampling_rate']
        self.samples_per_window = model_data['samples_per_window']
        self.stride = model_data['stride']
        
        print(f"RestModel loaded from {filepath}")
        return True
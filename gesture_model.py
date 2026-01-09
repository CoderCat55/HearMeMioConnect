# gesture_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

class GestureModel:
    def __init__(self, window_size_ms=100, sampling_rate=200):
        self.model = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=200)
        self.scaler = StandardScaler()
        self.window_size_ms = window_size_ms
        self.sampling_rate = sampling_rate
        self.gesture_labels = []

        # Calculate samples per window
        self.samples_per_window = int((window_size_ms / 1000) * sampling_rate)  # 20 samples
        self.stride = self.samples_per_window // 2  # 50% overlap = 10 samples stride

    @staticmethod
    def load_nested_data(base_path):
        """
        Yeni eklenen yardımcı fonksiyon:
        Klasör yapısını (base_path -> pX -> gesture_name -> *.npy) tarar.
        """
        gesture_data_dict = {}
        
        if not os.path.exists(base_path):
            print(f"Hata: {base_path} dizini bulunamadı!")
            return gesture_data_dict

        print(f"Veriler taranıyor: {base_path}...")
        
        # 1. Seviye: Katılımcı klasörleri (p1, p2...)
        for p_folder in os.listdir(base_path):
            p_path = os.path.join(base_path, p_folder)
            if os.path.isdir(p_path):
                # 2. Seviye: Jest klasörleri (yumruk, acik_el...)
                for gesture_name in os.listdir(p_path):
                    gesture_path = os.path.join(p_path, gesture_name)
                    if os.path.isdir(gesture_path):
                        # 3. Seviye: .npy dosyaları
                        for file in os.listdir(gesture_path):
                            if file.endswith(".npy"):
                                file_path = os.path.join(gesture_path, file)
                                try:
                                    data = np.load(file_path)
                                    if gesture_name not in gesture_data_dict:
                                        gesture_data_dict[gesture_name] = []
                                    gesture_data_dict[gesture_name].append(data)
                                except Exception as e:
                                    print(f"Dosya okunamadı {file}: {e}")
        
        return gesture_data_dict

    @staticmethod
    def extract_features(time_series_data):
        """
        Same feature extraction as before
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
                np.max(channel_data) - np.min(channel_data),
            ])
        return np.array(features)
    
    def train(self, gesture_data_dict):
        """
        Train on segmented gesture data from calibration_data/pXnew
        gesture_data_dict: {gesture_name: [list of numpy arrays]}
        """
        print("Extracting features from gesture data with sliding windows...")
        X = []
        y = []
        
        self.gesture_labels = sorted(gesture_data_dict.keys())
        
        for gesture_name in self.gesture_labels:
            samples = gesture_data_dict[gesture_name]
            for time_series in samples:
                # Slide window with 50% overlap
                num_windows = (len(time_series) - self.samples_per_window) // self.stride + 1
                
                for i in range(num_windows):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.samples_per_window
                    window = time_series[start_idx:end_idx]
                    
                    features = self.extract_features(window)
                    X.append(features)
                    y.append(gesture_name)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training GestureModel on {len(X)} windows from {len(self.gesture_labels)} gestures...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM (RandomForestClassifier used here)
        self.model.fit(X_scaled, y)
        
        print("GestureModel training complete!")
        return True
    
    def classify(self, features):
        """
        Classify a feature vector
        features: 1D numpy array (170 features)
        Returns: gesture name (string)
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0]
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'gesture_labels': self.gesture_labels,
            'window_size_ms': self.window_size_ms,
            'sampling_rate': self.sampling_rate,
            'samples_per_window': self.samples_per_window,
            'stride': self.stride
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"GestureModel saved to {filepath}")
    
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.gesture_labels = model_data['gesture_labels']
        self.window_size_ms = model_data['window_size_ms']
        self.sampling_rate = model_data['sampling_rate']
        self.samples_per_window = model_data['samples_per_window']
        self.stride = model_data['stride']
        
        print(f"GestureModel loaded from {filepath}")
        return True
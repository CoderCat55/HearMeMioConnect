# gesture_model.py
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier  # Changed from svm/RandomForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class PersonalModel:
    def __init__(self, window_size_samples, sampling_rate):
        # Changed to KNeighborsClassifier
        # n_neighbors=5 is a standard starting point; you can adjust this (e.g., 3 or 7)
        self.model = KNeighborsClassifier(n_neighbors=3, weights='distance') 
        
        self.scaler = StandardScaler()
        self.window_size_samples = window_size_samples
        self.sampling_rate = sampling_rate
        self.gesture_labels = []
        
        # Calculate samples per window
        self.samples_per_window = window_size_samples 
        self.stride = self.samples_per_window // 2  # 50% overlap
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
            # Mevcut Özellikler
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            range_val = max_val - min_val
            
            # YENİ: RMS (Root Mean Square) - Sinyal Gücü
            rms_val = np.sqrt(np.mean(channel_data**2))
            
            # YENİ: Zero Crossing Rate - Sinyal Hızı/Frekansı
            # Sinyalin sıfır noktasını kaç kez kestiğini sayar
            zero_crossings = np.where(np.diff(np.sign(channel_data)))[0].size / len(channel_data)
            
            features.extend([
                mean_val, std_val, min_val, max_val, 
                range_val, rms_val, zero_crossings
            ])
            
        return np.array(features)

    
    def train(self, gesture_data):
        X_train, y_train, X_test, y_test = [], [], [], []
    
        for label, samples in gesture_data.items():
            # Her hareketi kendi içinde dosya bazlı böl (Örn: 15 dosyanın 3'ü teste)
            split_idx = int(len(samples) * 0.8)
            train_samples = samples[:split_idx]
            test_samples = samples[split_idx:]
        
            # Eğitim verisi için pencereler oluştur
            for sample in train_samples:
                for i in range(0, len(sample) - self.samples_per_window + 1, self.stride):
                    # BU SATIRLAR İÇERİDE OLMALI
                    X_train.append(self.extract_features(sample[i:i + self.samples_per_window]))
                    y_train.append(label)
                
            # Test verisi için pencereler oluştur (Tamamen farklı dosyalardan)
            for sample in test_samples:
                for i in range(0, len(sample) - self.samples_per_window + 1, self.stride):
                    # BU SATIRLAR İÇERİDE OLMALI
                    X_test.append(self.extract_features(sample[i:i + self.samples_per_window]))
                    y_test.append(label)

        if len(X_train) == 0:
            print("Training data is empty!")
            return

        # Eğitim ve test setlerini oluştur
        X_train_scaled = self.scaler.fit_transform(np.array(X_train))
        self.model.fit(X_train_scaled, y_train)

        # Accuracy ölçümü (İsteğe bağlı)
        if len(X_test) > 0:
            X_test_scaled = self.scaler.transform(np.array(X_test))
            y_pred = self.model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            print(f"Personal Model Accuracy: %{acc * 100:.2f}")

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
            'window_size_samples': self.window_size_samples,  # Changed from window_size_ms
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
        self.window_size_samples = model_data['window_size_samples'] 
        self.sampling_rate = model_data['sampling_rate']
        self.samples_per_window = model_data['samples_per_window']
        self.stride = model_data['stride']
        
        print(f"GestureModel loaded from {filepath}")
        return True
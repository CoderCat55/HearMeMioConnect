from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pickle
import os
import glob

class RestDetector:
    # binary SVM for understanding if data is rest or not
    def __init__(self, window_size):  # samples, not ms
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
        self.window_size = window_size

    @staticmethod
    def extract_features(time_series_data):
        """
        IDENTICAL to gesture_model.extract_features()
        time_series_data shape: (time_steps, 34)
        Returns: 1D feature vector (170 features = 34 channels × 5 features)
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
        
    def train(self):
        """
        Train binary SVM on ALL participant data
        - rest class: files starting with "rest" in rows_deleted/p{1-6}
        - non-rest class: files NOT starting with "rest" in rows_deleted/p{1-6}
        Uses sliding windows with 50% overlap (stride = window_size // 2)
        """
        print("Loading rest data from rows_deleted/p{1-6}...")
        rest_samples = []
        for participant_id in range(1, 7):
            folder = f'rows_deleted/p{participant_id}'
            if os.path.exists(folder):
                files = glob.glob(f'{folder}/*.npy')
                # FILTER: only files starting with "rest"
                files = [f for f in files if os.path.basename(f).startswith('rest')]
                for file in files:
                    data = np.load(file)
                    rest_samples.append(data)
                    print(f"  ✓ Loaded Rest: {os.path.basename(file)} (shape: {data.shape})")
        
        print(f"\nLoading non-rest data from rows_deleted/p{{1-6}}...")
        not_rest_samples = []
        for participant_id in range(1, 7):
            folder = f'rows_deleted/p{participant_id}'
            if os.path.exists(folder):
                files = glob.glob(f'{folder}/*.npy') 
                # FILTER: only files NOT starting with "rest"
                files = [f for f in files if not os.path.basename(f).startswith('rest')]
                for file in files:
                    data = np.load(file)
                    not_rest_samples.append(data)
                    print(f"  ✓ Loaded Non-Rest: {os.path.basename(file)} (shape: {data.shape})")
        
        if len(rest_samples) == 0 or len(not_rest_samples) == 0:
            print("ERROR: Need both rest and non-rest data!")
            return False
        
        print(f"\nExtracting features with sliding windows (window_size={self.window_size}, stride={self.window_size//2})...")
        
        X = []
        y = []
        stride = self.window_size // 2  # 50% overlap
        
        # Process rest samples (label = 0)
        for time_series in rest_samples:
            num_windows = (len(time_series) - self.window_size) // stride + 1
            if num_windows <= 0: continue
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + self.window_size
                window = time_series[start_idx:end_idx]
                X.append(self.extract_features(window))
                y.append(0) 
        
        # Process non-rest samples (label = 1)
        for time_series in not_rest_samples:
            num_windows = (len(time_series) - self.window_size) // stride + 1
            if num_windows <= 0: continue
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + self.window_size
                window = time_series[start_idx:end_idx]
                X.append(self.extract_features(window))
                y.append(1)
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # --- NEW EVALUATION LOGIC ---
        # Split into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining binary SVM on {len(X_train)} windows:")
        print(f"  - Training Rest samples: {np.sum(y_train == 0)}")
        print(f"  - Training Non-rest samples: {np.sum(y_train == 1)}")
        
        # Train SVM
        self.model.fit(X_train, y_train)
        
        # Performance Evaluation
        y_pred = self.model.predict(X_test)
        
        print("\n" + "="*40)
        print("      REST DETECTOR PERFORMANCE")
        print("="*40)
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        

        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Rest', 'Non-Rest']))
        print("="*40)
        
        print("✓ RestDetector training and evaluation complete!")
        return True

    def predict(self, window_data):
        """
        Predict if window is rest or not-rest
        Returns: True if rest (0), False if non-rest (1)
        """
        features = self.extract_features(window_data)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0] == 0

    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'window_size': self.window_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"RestDetector saved to {filepath}")

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.window_size = model_data['window_size']
        print(f"RestDetector loaded from {filepath}")
        return True
    
    if __name__ == "__main__":
        train()
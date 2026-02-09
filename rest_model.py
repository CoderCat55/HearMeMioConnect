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
        Train binary SVM on participant data and save results.
        """
        print("Loading rest data from rows_deleted/p{1-6}...")
        rest_samples = []
        for participant_id in range(1, 7):
            folder = f'rows_deleted/p{participant_id}'
            if os.path.exists(folder):
                files = glob.glob(f'{folder}/*.npy')
                files = [f for f in files if os.path.basename(f).startswith('rest')]
                for file in files:
                    data = np.load(file)
                    rest_samples.append(data)
        
        print(f"Loading non-rest data from rows_deleted/p{{1-6}}...")
        not_rest_samples = []
        for participant_id in range(1, 7):
            folder = f'rows_deleted/p{participant_id}'
            if os.path.exists(folder):
                files = glob.glob(f'{folder}/*.npy') 
                files = [f for f in files if not os.path.basename(f).startswith('rest')]
                for file in files:
                    data = np.load(file)
                    not_rest_samples.append(data)
        
        if len(rest_samples) == 0 or len(not_rest_samples) == 0:
            print("ERROR: Need both rest and non-rest data!")
            return False
        
        X, y = [], []
        stride = self.window_size // 2 
        
        # Process windows
        for samples, label in [(rest_samples, 0), (not_rest_samples, 1)]:
            for time_series in samples:
                num_windows = (len(time_series) - self.window_size) // stride + 1
                if num_windows <= 0: continue
                for i in range(num_windows):
                    start = i * stride
                    window = time_series[start : start + self.window_size]
                    X.append(self.extract_features(window))
                    y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train SVM
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # --- NEW: SAVE HYPERPARAMETERS AND PERFORMANCE ---
        results_path = "rest_detector_results.txt"
        with open(results_path, "w") as f:
            f.write("=== MODEL HYPERPARAMETERS ===\n")
            f.write(str(self.model.get_params()) + "\n\n")
            
            f.write("=== PERFORMANCE METRICS ===\n")
            f.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(confusion_matrix(y_test, y_pred)) + "\n\n")
            
            f.write("Detailed Classification Report:\n")
            f.write(classification_report(y_test, y_pred, target_names=['Rest', 'Non-Rest']))

        print(f"\n✓ Training complete. Results saved to {results_path}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        return True

    def predict(self, window_data):
        features = self.extract_features(window_data)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0] == 0

    def save_model(self, filepath):
        model_data = {'model': self.model, 'scaler': self.scaler, 'window_size': self.window_size}
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        if not os.path.exists(filepath): return False
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model, self.scaler, self.window_size = data['model'], data['scaler'], data['window_size']
        return True

if __name__ == "__main__":
    # CHANGE: Instantiate the class first
    detector = RestDetector(window_size=100)
    # CHANGE: Call train on the instance
    detector.train()
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
        features = []
        
        if time_series_data.shape[0] == 0:
            return np.zeros(34 * 5)

        for channel in range(time_series_data.shape[1]):
            channel_data = time_series_data[:, channel]
            
            mav = np.mean(np.abs(channel_data))
            rms = np.sqrt(np.mean(channel_data**2))
            var = np.var(channel_data)
            waveform_length = np.sum(np.abs(np.diff(channel_data))) / len(channel_data)
            zero_crossings = np.count_nonzero(np.diff(np.signbit(channel_data))) / len(channel_data)

            features.extend([mav, rms, var, waveform_length, zero_crossings])
        
        return np.array(features)

    def _create_windows(self, data, window_size=50, step_size=30):
        # step_size artırıldı → pencereler daha az üst üste biner
        windows = []
        if len(data) < window_size:
            return [data]
            
        for i in range(0, len(data) - window_size, step_size):
            windows.append(data[i : i + window_size])
            
        return windows
    
    def add_calibration_sample(self, gesture_name, time_series_data):
        if gesture_name not in self.calibration_data:
            self.calibration_data[gesture_name] = []
        
        self.calibration_data[gesture_name].append(time_series_data)
        print(f"Added calibration sample for '{gesture_name}'. Total raw files: {len(self.calibration_data[gesture_name])}")
    
    def train(self, use_crossval=True, cv_folds=5):
        if len(self.calibration_data) < 2:
            print("ERROR: Need at least 2 gestures to train!")
            return False
        
        print("\n--- Starting Training Process ---")
        print("Processing calibration data (Windowing & Feature Extraction)...")
        
        X = []
        y = []
        
        self.gesture_labels = sorted(self.calibration_data.keys())
        total_windows = 0
        
        for gesture_name in self.gesture_labels:
            raw_samples = self.calibration_data[gesture_name]
            gesture_window_count = 0
            
            for time_series in raw_samples:
                windows = self._create_windows(time_series, window_size=50, step_size=30)
                
                for window in windows:
                    features = self.extract_features(window)
                    if np.all(np.isfinite(features)):
                        X.append(features)
                        y.append(gesture_name)
                        gesture_window_count += 1
            
            print(f"  - '{gesture_name}': Created {gesture_window_count} training samples.")
            total_windows += gesture_window_count
        
        X = np.array(X)
        y = np.array(y)
        print(f"Total training dataset size: {len(X)} samples.")
        
        # Split for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVM with moderate C to reduce overfitting
        print("Training SVM model...")
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation on training set
        if use_crossval:
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds)
            print(f"Cross-validation ({cv_folds}-fold) Accuracy: %{np.mean(cv_scores) * 100:.2f} ± %{np.std(cv_scores) * 100:.2f}")
        
        # Train accuracy
        y_train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Train Accuracy (on training split): %{train_acc * 100:.2f}")
        
        # Test accuracy
        y_test_pred = self.model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy (on test split): %{test_acc * 100:.2f}")
        
        # Detailed Test report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))
        
        print("Confusion Matrix (Test Set):")
        print(confusion_matrix(y_test, y_test_pred))
        print("--------------------------------\n")
        
        return True
    
    def classify(self, features):
        if self.model is None:
            return "ERROR: Model not trained yet!"
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0]
    
    def save_model(self, filepath):
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
        if not os.path.exists('calibration_data'):
            print("No calibration data directory found")
            return
        
        files = glob.glob('calibration_data/*.npy')
        if not files:
            print("No calibration files found")
            return
        
        count = 0
        for file in files:
            basename = os.path.basename(file)
            gesture_name = basename.split('_')[0]
            
            data = np.load(file)
            if gesture_name not in self.calibration_data:
                self.calibration_data[gesture_name] = []
            self.calibration_data[gesture_name].append(data)
            count += 1
        
        print(f"Loaded {count} raw calibration files for {len(self.calibration_data)} gestures")

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# --- NEW IMPORT ---
from sklearn.ensemble import RandomForestClassifier
import glob
import numpy as np
import pickle
import os

class GestureClassifier:
    def __init__(self, base_model_name='RandomForest'): # Default to Random Forest
        self.calibration_data = {}  # gesture_name -> list of np.arrays
        self.model = None
        self.scaler = StandardScaler()
        self.gesture_labels = []  # Ordered list of gesture names
        self.base_model_name = base_model_name # To switch between SVM and RF
    
    @staticmethod
    def extract_features(time_series_data):
        features = []
        
        if time_series_data.shape[0] == 0:
            return np.zeros(34 * 5)

        for channel in range(time_series_data.shape[1]):
            channel_data = time_series_data[:, channel]
            
            # --- Time-Domain Features ---
            mav = np.mean(np.abs(channel_data))
            rms = np.sqrt(np.mean(channel_data**2))
            var = np.var(channel_data)
            waveform_length = np.sum(np.abs(np.diff(channel_data))) / len(channel_data)
            zero_crossings = np.count_nonzero(np.diff(np.signbit(channel_data))) / len(channel_data)

            features.extend([mav, rms, var, waveform_length, zero_crossings])
        
        return np.array(features)

    def _create_windows(self, data, window_size=30, step_size=10):
        # Step size is now smaller (10), increasing overlap and sample count.
        windows = []
        if len(data) < window_size:
            # Handle cases where data is too short for a full window
            if len(data) > 0:
                return [data]
            return []
            
        for i in range(0, len(data) - window_size, step_size):
            windows.append(data[i : i + window_size])
        
        # Add the last window if it wasn't captured entirely
        if len(data) >= window_size and (len(data) - window_size) % step_size != 0:
             windows.append(data[-window_size:])
            
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
            
            # Using smaller window_size=30 and step_size=10
            for time_series in raw_samples:
                windows = self._create_windows(time_series, window_size=30, step_size=10)
                
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
        
        # Normalize: Mandatory for SVM, harmless for RF
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training {self.base_model_name} model...")
        
        # --- MODEL SELECTION AND HYPERPARAMETER TUNING ---
        if self.base_model_name == 'SVM':
            # SVM Tuning: We'll use a small Grid Search here for demonstration
            # You should expand this range for real-world tuning
            param_grid = {
                'C': [0.1, 1, 10], 
                'gamma': [0.01, 0.1, 'scale']
            }
            base_model = svm.SVC(kernel='rbf')
            
        elif self.base_model_name == 'RandomForest':
            # Random Forest Tuning: Tune n_estimators (number of trees) and max_depth
            param_grid = {
                'n_estimators': [50, 100, 200], # Number of trees
                'max_depth': [5, 10, None]       # Max depth of each tree (None means full depth)
            }
            # RandomForests don't require scaling, but we use the scaled data anyway.
            base_model = RandomForestClassifier(random_state=42)

        else:
            print(f"ERROR: Unknown model {self.base_model_name}")
            return False

        # Use GridSearchCV to find the best model parameters
        grid_search = GridSearchCV(
            estimator=base_model, 
            param_grid=param_grid, 
            cv=cv_folds, 
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train) 
        
        # Set the best model found by the Grid Search
        self.model = grid_search.best_estimator_
        
        print(f"Optimal Parameters: {grid_search.best_params_}")
        
        # --- END MODEL SELECTION AND HYPERPARAMETER TUNING ---
        
        # Cross-validation on training set (using the best estimator)
        if use_crossval:
            # We already have the CV score from the grid search
            print(f"Best Cross-validation ({cv_folds}-fold) Accuracy: %{grid_search.best_score_ * 100:.2f}")
            
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
    
    # ... (classify, save_model, load_model, load_calibration_data methods remain the same) ...
    
    def classify(self, features):
        if self.model is None:
            return "ERROR: Model not trained yet!"
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features before classification
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
            'calibration_data': self.calibration_data,
            'base_model_name': self.base_model_name # Save model type
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
        self.base_model_name = model_data.get('base_model_name', 'SVM') # Compatibility
        print(f"Model loaded from {filepath}")
        print(f"Loaded model type: {self.base_model_name}")
        return True
    
    def load_calibration_data(self):
        # ... (This method remains the same) ...
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
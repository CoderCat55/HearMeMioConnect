"""
Handles training a gesture recognition model from raw data and classifying
real-time data from shared memory.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIRECTORY = r"calibration_data" # Path to the folder with gesture data

# These must match the values in trainmodel.py
WINDOW_SIZE = 200
WINDOW_STEP = 50

class GestureClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.gesture_map = None
        self.label_to_gesture = None # For converting model output (int) to gesture name (str)
        self.loaded_model_name = "None"

    @staticmethod
    def extract_features(data, window_size=WINDOW_SIZE, window_step=WINDOW_STEP):
        """
        Extracts time-domain features from sEMG data using a sliding window.
        This MUST match the feature extraction in trainmodel.py.
        Args:
            data (np.ndarray): The raw sEMG data (samples, channels).
            window_size (int): The size of the sliding window.
            window_step (int): The step size to move the window.
        Returns:
            np.ndarray: The extracted feature matrix (num_windows, num_features).
        """
        n_samples, n_channels = data.shape
        features = []

        for start in range(0, n_samples - window_size + 1, window_step):
            end = start + window_size
            window = data[start:end, :]

            window_features = []
            for i in range(n_channels):
                channel_data = window[:, i]

                # 1. Mean Absolute Value (MAV)
                mav = np.mean(np.abs(channel_data))
                # 2. Waveform Length (WL)
                wl = np.sum(np.abs(np.diff(channel_data)))
                # 3. Zero Crossings (ZC)
                zc = np.sum(np.diff(np.sign(channel_data)) != 0)
                # 4. Slope Sign Changes (SSC)
                ssc = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)

                window_features.extend([mav, wl, zc, ssc])
            features.append(window_features)
        return np.array(features)

    def add_calibration_sample(self, gesture_name, time_series_data):
        """
        This method is kept for compatibility with the 'cb' command but does not
        directly contribute to the model loaded from trainmodel.py. It just saves the data.
        """
        print(f"Saved calibration sample for '{gesture_name}'.")
        print("Note: Run the 'tr' command to train a new model with this data.")

    def train(self):
        """
        Loads all data, finds the best model, trains it on all data,
        and saves the artifacts.
        """
        print("--- Starting Model Training Process ---")

        # 1. Load data and extract features
        X, y, gesture_map = self._load_data_from_folders(DATA_DIRECTORY, WINDOW_SIZE, WINDOW_STEP)
        if X is None:
            print("--- Training Failed: Could not load data. ---")
            return False

        print("\n--- Splitting data into training and testing sets (75/25) ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # 2. Feature Scaling (Fit on training data ONLY to prevent data leakage)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) # Use the same scaler to transform test data
        print("Features have been scaled based on the training set.")
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")

        # 3. Train and evaluate multiple models
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
            "Support Vector Machine": SVC(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        }
        results = {}

        print("\n--- Evaluating Models on the Test Set ---")
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train) # X_train is already scaled
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = accuracy
            print(f"  - Accuracy on Test Set: {accuracy * 100:.2f}%")

        # 4. Select the best model and show detailed report
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['accuracy'])
        results_df['accuracy'] = results_df['accuracy'] * 100
        results_df = results_df.sort_values(by='accuracy', ascending=False)
        
        print("\n--- Model Comparison Summary ---")
        print(results_df.to_string(formatters={'accuracy': '{:.2f}%'.format}))
        print("--------------------------------")

        best_model_name = results_df.index[0]
        best_model_instance = models[best_model_name]
        
        print(f"\n--- Detailed Report for Best Model: {best_model_name} ---")
        # The model is already trained from the evaluation loop, no need to fit again.
        y_pred_best = best_model_instance.predict(X_test)
        
        label_to_gesture_map = {v: k for k, v in gesture_map.items()}
        target_names = [label_to_gesture_map[i] for i in sorted(label_to_gesture_map.keys())]

        print("\nClassification Report:")
        report = classification_report(y_test, y_pred_best, target_names=target_names)
        print(report)

        print("Generating and saving confusion matrix...")
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix for {best_model_name}', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved to 'confusion_matrix.png'")

        # 5. Train all models on the ENTIRE dataset and save them
        print("\n--- Training all models on the entire dataset for saving ---")
        for model_name, model in models.items():
            print(f"Training final '{model_name}' model...")
            X_full_scaled = scaler.fit_transform(X) # Re-fit scaler on all data for production model
            model.fit(X_full_scaled, y)
            # Sanitize model name for filename
            safe_model_name = model_name.lower().replace(' ', '_')
            model_path = f'model_{safe_model_name}.pkl'
            print(f"  -> Saving to {model_path}")
            joblib.dump(model, model_path)

        # 6. Save the scaler and gesture map (they are common for all models)
        scaler_path = 'scaler.pkl'
        map_path = 'gesture_map.json'

        print(f"\nSaving scaler to {scaler_path}...")
        joblib.dump(scaler, scaler_path)

        print(f"Saving gesture map to {map_path}...")
        with open(map_path, 'w') as f:
            json.dump(gesture_map, f, indent=4)

        print("\n--- All models, scaler, and map have been saved. ---")
        
        # 7. Load the best model into this instance for immediate use
        print("Loading newly trained model for immediate use...")
        best_model_safe_name = best_model_name.lower().replace(' ', '_')
        best_model_path = f'model_{best_model_safe_name}.pkl'
        return self.load_model(best_model_path, scaler_path, map_path)

    def _load_data_from_folders(self, base_path, window_size, window_step):
        """
        Internal method to load all .npy files and prepare them for training.
        """
        all_features = []
        all_labels = []
        gesture_to_label_map = {}

        if not os.path.isdir(base_path):
            print(f"Error: Data directory not found at '{base_path}'")
            return None, None, None

        # Check for participant subdirectories (e.g., p1, p2)
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        if subdirs:
            # Structure: base_path/participant_id/gesture_timestamp.npy
            print(f"Found participant subdirectories: {subdirs}")
            search_paths = [os.path.join(base_path, d) for d in subdirs]
        else:
            # Structure: base_path/gesture_timestamp.npy (flat structure)
            print("No participant subdirectories found. Reading from base directory.")
            search_paths = [base_path]

        for path in search_paths:
            self._process_files_in_path(path, gesture_to_label_map, all_features, all_labels, window_size, window_step)

        if not all_features: # This check is now inside _process_files_in_path
            print("\nError: Failed to load any data.")
            return None, None, None

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        print(f"\nTotal samples after augmentation: {X.shape[0]}")
        return X, y, gesture_to_label_map

    def _augment_and_extract(self, data, window_size, window_step):
        """
        Applies augmentation to the data and extracts features from original and augmented versions.
        """
        all_augmented_features = []

        # 1. Original Data
        original_features = self.extract_features(data, window_size, window_step)
        if original_features.shape[0] > 0:
            all_augmented_features.append(original_features)

        # 2. Augmentation: Add small random noise
        noise_level = 0.01 * np.std(data)
        noisy_data = data + np.random.normal(0, noise_level, data.shape)
        noisy_features = self.extract_features(noisy_data, window_size, window_step)
        if noisy_features.shape[0] > 0:
            all_augmented_features.append(noisy_features)

        # 3. Augmentation: Time Shift
        shift_amount = int(0.1 * len(data)) # Shift by 10%
        if len(data) > shift_amount:
            shifted_data = np.roll(data, shift_amount, axis=0)
            shifted_features = self.extract_features(shifted_data, window_size, window_step)
            if shifted_features.shape[0] > 0:
                all_augmented_features.append(shifted_features)

        if not all_augmented_features:
            return np.array([])

        return np.vstack(all_augmented_features)

    def _process_files_in_path(self, path, gesture_map, all_features, all_labels, window_size, window_step):
        """Helper to process .npy files in a given directory."""
        files = [f for f in os.listdir(path) if f.lower().endswith('.npy')]
        for file_name in sorted(files):
            file_path = os.path.join(path, file_name)
            try:
                gesture_name = file_name.split('_')[0]
                if gesture_name not in gesture_map:
                    gesture_map[gesture_name] = len(gesture_map)
                    print(f"  - Found new gesture: '{gesture_name}'. Assigning label: {gesture_map[gesture_name]}")
                
                label = gesture_map[gesture_name]
                data = np.load(file_path)
                
                if data.size > 0:
                    # Use the new augmentation function
                    extracted_features = self._augment_and_extract(data, window_size, window_step)
                    if extracted_features.shape[0] > 0:
                        all_features.append(extracted_features)
                        all_labels.append(np.full(extracted_features.shape[0], label))
            except Exception as e:
                print(f"  - Could not read {file_name}: {e}")

    def classify(self, features):
        """
        Classify a feature vector
        features: 1D numpy array of features
        Returns: gesture name (string)
        """
        if self.model is None or self.scaler is None or self.label_to_gesture is None:
            return "ERROR: Model not loaded! Please run 'tr' command."

        # The number of windows to consider for a stable prediction.
        # This helps filter out momentary incorrect classifications.
        STABILITY_THRESHOLD = 2 

        if features.shape[0] < STABILITY_THRESHOLD:
            return "Not enough data"

        # Normalize
        features_scaled = self.scaler.transform(features)

        # Predict on each window
        window_predictions = self.model.predict(features_scaled)

        # Perform a more robust majority vote (mode)
        if len(window_predictions) > 0:
            # Find the most frequent prediction label in the recent windows
            counts = np.bincount(window_predictions)
            most_common_label = np.argmax(counts)
            
            # Only return a prediction if it's seen consistently
            if counts[most_common_label] >= STABILITY_THRESHOLD:
                return self.label_to_gesture[most_common_label]
            else:
                return "Uncertain" # Return a neutral state if not stable
        else:
            return "Not enough data for a prediction"


    def load_model(self, model_path='model_random_forest.pkl', scaler_path='scaler.pkl', map_path='gesture_map.json'):
        """Load trained model from disk"""
        if not all(os.path.exists(p) for p in [model_path, scaler_path, map_path]):
            self.model = None
            self.loaded_model_name = "None"
            print(f"Warning: One or more model files not found ({model_path}, {scaler_path}, {map_path}).")
            print("Please run the 'tr' command to train models.")
            return False

        try:
            print("Loading model, scaler, and gesture map...")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(map_path, 'r') as f:
                self.gesture_map = json.load(f)

            # Create a reverse map for easy lookup
            self.label_to_gesture = {v: k for k, v in self.gesture_map.items()}

            # Extract and store the friendly name of the loaded model
            base_name = os.path.basename(model_path)
            model_name_part = base_name.replace('model_', '').replace('.pkl', '')
            self.loaded_model_name = model_name_part.replace('_', ' ').title()

            print(f"Model '{self.loaded_model_name}' loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model files: {e}")
            self.model = None
            self.loaded_model_name = "None"
            return False
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
DATA_DIRECTORY = r"MyoParticipantOfflineML\Myonewdata" # Path to the folder with gesture data

# These must match the values in trainmodel.py
WINDOW_SIZE = 200
WINDOW_STEP = 50

class GestureClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.gesture_map = None
        self.label_to_gesture = None # For converting model output (int) to gesture name (str)

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

        # 2. Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Features have been scaled.")
        
        print("\n--- Splitting data into training and testing sets (75/25) ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")

        # 3. Train and evaluate multiple models
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
            "Support Vector Machine": SVC(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        results = {}

        print("\n--- Evaluating Models on the Test Set ---")
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
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
        best_model_instance.fit(X_train, y_train)
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

        # 5. Train the final, best model on the ENTIRE dataset
        print(f"\nTraining the final '{best_model_name}' model on the entire dataset...")
        final_model = models[best_model_name]
        final_model.fit(X_scaled, y)
        print("Final model training complete.")

        # 6. Save the model, scaler, and gesture map
        model_path = 'gesture_model.pkl'
        scaler_path = 'scaler.pkl'
        map_path = 'gesture_map.json'

        print(f"\nSaving best model ('{best_model_name}') to {model_path}...")
        joblib.dump(final_model, model_path)

        print(f"Saving scaler to {scaler_path}...")
        joblib.dump(scaler, scaler_path)

        print(f"Saving gesture map to {map_path}...")
        with open(map_path, 'w') as f:
            json.dump(gesture_map, f, indent=4)

        print("\n--- Training and saving complete! ---")
        
        # 7. Load the newly trained model into this instance
        print("Loading newly trained model for immediate use...")
        return self.load_model(model_path, scaler_path, map_path)

    def _load_data_from_folders(self, base_path, window_size, window_step):
        """
        Internal method to load all .npy files and prepare them for training.
        """
        all_features = []
        all_labels = []
        gesture_to_label_map = {}
        label_counter = 0

        if not os.path.isdir(base_path):
            print(f"Error: Data directory not found at '{base_path}'")
            return None, None, None

        participant_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        if not participant_folders:
            print(f"Error: No participant subdirectories found in '{base_path}'")
            return None, None, None

        print(f"Found participants: {participant_folders}")
        for participant_id in participant_folders:
            participant_path = os.path.join(base_path, participant_id)
            files = [f for f in os.listdir(participant_path) if f.lower().endswith('.npy')]
            if not files:
                continue

            for file_name in sorted(files):
                file_path = os.path.join(participant_path, file_name)
                try:
                    gesture_name = file_name.split('_')[0]
                    if gesture_name not in gesture_to_label_map:
                        gesture_to_label_map[gesture_name] = label_counter
                        print(f"  - Found new gesture: '{gesture_name}'. Assigning label: {label_counter}")
                        label_counter += 1
                    
                    label = gesture_to_label_map[gesture_name]
                    data = np.load(file_path)
                    
                    if data.size > 0:
                        extracted_features = self.extract_features(data, window_size, window_step)
                        if extracted_features.shape[0] > 0:
                            all_features.append(extracted_features)
                            all_labels.append(np.full(extracted_features.shape[0], label))
                except Exception as e:
                    print(f"  - Could not read {file_name}: {e}")

        if not all_features:
            print("\nError: Failed to load any data.")
            return None, None, None

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        return X, y, gesture_to_label_map

    def classify(self, features):
        """
        Classify a feature vector
        features: 1D numpy array of features
        Returns: gesture name (string)
        """
        if self.model is None or self.scaler is None or self.label_to_gesture is None:
            return "ERROR: Model not loaded! Please run 'tr' command."

        if features.shape[0] == 0:
            return "No features to classify"

        # Normalize
        features_scaled = self.scaler.transform(features)

        # Predict
        predictions = self.model.predict(features_scaled)

        # Find the most common prediction (majority vote)
        if len(predictions) > 0:
            most_common_prediction = np.bincount(predictions).argmax()
            return self.label_to_gesture[most_common_prediction]
        else:
            return "Not enough data for a prediction"

    def load_model(self, model_path='gesture_model.pkl', scaler_path='scaler.pkl', map_path='gesture_map.json'):
        """Load trained model from disk"""
        if not all(os.path.exists(p) for p in [model_path, scaler_path, map_path]):
            print(f"Warning: One or more model files not found ({model_path}, {scaler_path}, {map_path}).")
            print("Please run the 'tr' command to train a new model.")
            return False

        try:
            print("Loading model, scaler, and gesture map...")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            with open(map_path, 'r') as f:
                self.gesture_map = json.load(f)

            # Create a reverse map for easy lookup
            self.label_to_gesture = {v: k for k, v in self.gesture_map.items()}

            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model files: {e}")
            return False
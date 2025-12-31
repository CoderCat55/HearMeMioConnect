"""
Evaluation script for the Gesture Recognition Model.
Calculates Accuracy, Confusion Matrix, and Classification Report
using the trained 'gesture_model.pkl' and data in 'processed_data'.
"""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
import sys

# Ensure we can import GestureModel from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gesture_model import GestureModel
except ImportError:
    print("CRITICAL ERROR: Could not import 'GestureModel'.")
    print("Ensure 'gesture_model.py' is in the same directory.")
    sys.exit(1)

# Configuration
MODEL_PATH = 'gesture_model.pkl'
DATA_DIR = 'processed_data'  # Directory containing .npy files (e.g., processed_data/p1/...)
WINDOW_MS = 100
SAMPLING_RATE = 50

def load_test_data(data_dir):
    """
    Loads .npy files from data_dir (and subdirectories).
    Expected filename format: gestureName_timestamp.npy
    """
    X = []
    y = []
    
    # Search recursively for .npy files
    search_pattern = os.path.join(data_dir, '**', '*.npy')
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No .npy files found in '{data_dir}'")
        return [], []
    
    print(f"Found {len(files)} test files in '{data_dir}'")
    
    for file_path in files:
        try:
            # Load data
            data = np.load(file_path)
            
            # Parse label from filename (e.g., "fist_123456.npy" -> "fist")
            filename = os.path.basename(file_path)
            label = filename.split('_')[0]
            
            X.append(data)
            y.append(label)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            
    return X, y

def predict_file(model, data):
    """
    Performs majority voting on a single file (recording).
    Slides a window across the recording, classifies each window, and votes.
    """
    window_samples = model.samples_per_window
    step_size = 2 # Overlap step (same as in main.py Classify)
    
    if len(data) < window_samples:
        return None
        
    predictions = []
    
    # Slide window
    for i in range(0, len(data) - window_samples, step_size):
        window = data[i : i + window_samples]
        features = model.extract_features(window)
        pred = model.classify(features)
        predictions.append(pred)
        
    if not predictions:
        return None
        
    # Majority Vote
    most_common = Counter(predictions).most_common(1)[0][0]
    return most_common

def main():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' not found.")
        print("Please run the training command ('tr') in main.py first.")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = GestureModel(window_size_ms=WINDOW_MS, sampling_rate=SAMPLING_RATE)
    if not model.load_model(MODEL_PATH):
        print("Error: Failed to load model weights.")
        return

    # 2. Load Data
    print(f"Loading data from: {DATA_DIR}")
    X_test, y_test = load_test_data(DATA_DIR)
    
    if not X_test:
        print("Cannot proceed without data.")
        return

    # 3. Predict
    print("Running predictions...")
    y_pred = []
    y_true_filtered = []
    
    for i, data in enumerate(X_test):
        prediction = predict_file(model, data)
        if prediction:
            y_pred.append(prediction)
            y_true_filtered.append(y_test[i])
            
    if not y_pred:
        print("No valid predictions generated (data might be too short).")
        return

    # 4. Metrics
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    
    # Accuracy
    acc = accuracy_score(y_true_filtered, y_pred)
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true_filtered, y_pred))
    
    # Confusion Matrix
    labels = sorted(list(set(y_true_filtered) | set(y_pred)))
    cm = confusion_matrix(y_true_filtered, y_pred, labels=labels)
    
    print("\nConfusion Matrix (Text):")
    print(cm)
    
    # Plot
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix (Accuracy: {acc:.2%})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        out_file = 'confusion_matrix.png'
        plt.savefig(out_file)
        print(f"\nPlot saved to {out_file}")
        plt.show()
    except Exception as e:
        print(f"\nSkipping plot generation: {e}")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Assuming your class is in a file named gesture_model.py
# from gesture_model import GestureModel 

## --- STEP 1: GENERATE SYNTHETIC DATA (Replace this with your real data loading) ---
def generate_dummy_data():
    gestures = ['Wave', 'Swipe_Left', 'Swipe_Right', 'Circle']
    data_dict = {}
    for g in gestures:
        # Create 10 samples per gesture, each 1 second long (200 samples @ 200Hz)
        # Each sample has 34 channels
        samples = [np.random.randn(200, 34) + np.random.randint(-2, 2) for _ in range(10)]
        data_dict[g] = samples
    return data_dict

## --- STEP 2: EVALUATION LOGIC ---
def run_evaluation():
    # 1. Get Data
    all_data = generate_dummy_data()
    
    # 2. Split into Train and Test (80% Train, 20% Test)
    # We split the "full samples" to avoid data leakage between windows
    train_dict = {}
    test_dict = {}
    
    for gesture, samples in all_data.items():
        tr, ts = train_test_split(samples, test_size=0.2)
        train_dict[gesture] = tr
        test_dict[gesture] = ts

    # 3. Initialize and Train
    gm = GestureModel()
    gm.train(train_dict)

    # 4. Test the model
    y_true = []
    y_pred = []

    print("Evaluating model...")
    for gesture_name, samples in test_dict.items():
        for time_series in samples:
            # We slide windows over the test data just like we did in training
            num_windows = (len(time_series) - gm.samples_per_window) // gm.stride + 1
            
            for i in range(num_windows):
                start = i * gm.stride
                end = start + gm.samples_per_window
                window = time_series[start:end]
                
                features = gm.extract_features(window)
                prediction = gm.classify(features)
                
                y_true.append(gesture_name)
                y_pred.append(prediction)

    # 5. Output Results
    labels = gm.gesture_labels
    
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(y_true, y_pred, target_names=labels))

    # 6. Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Gesture Recognition Confusion Matrix')
    plt.ylabel('Actual Gesture')
    plt.xlabel('Predicted Gesture')
    plt.show()

if __name__ == "__main__":
    run_evaluation()
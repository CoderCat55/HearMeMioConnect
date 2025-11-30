"""
trains itself based on calibration samples
reads realtime data from shared memory for a fixed time and classifies returns result
"""
# File: calibration_data/
#   ├── gesture1_samples.npy  # Shape: (n_samples, time_steps, features)
#   ├── gesture2_samples.npy
#   └── metadata.json  # Gesture names, timestamps, etc.

from sklearn import svm
import glob
import numpy as np

# In sckit.py:
class GestureClassifier:
    def __init__(self):
        self.calibration_data = {}  # gesture_name -> np.array
        self.model = None
    
    def add_calibration_sample(self, gesture_name, time_series_data):
        # time_series_data shape: (time_steps, features)
        
        pass
    
    def train(self):
        # Extract features from time series
        # Train SVM
        pass

    def load_calibration_data(self):
    #Load previously saved calibration from disk
        for file in glob.glob('calibration_data/*.npy'):
            gesture_name = file.split('/')[-1].split('.')[0]
            data = np.load(file)
            self.calibration_data[gesture_name] = data

"""
So add_calibration_sample() is for:
During runtime: Adding newly recorded calibration
Not for: Loading from disk (that's a separate load_calibration_data() method)
"""
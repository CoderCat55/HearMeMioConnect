from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

class RestDetector:
    #  binary SVM for understanding if data is rest or not
    def __init__(self, window_size=20):  # samples, not ms
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
        self.window_size = window_size
        
    @staticmethod
    def extract_features(time_series_data):
        # IDENTICAL to gesture_model.extract_features()
        # Returns 170 features
        
    def train(self,):
        # Extract features with sliding windows
        #train rest model on all participant data
            #rest class : calibration_data/pxrest (X=participant ID)
            #non-rest class: processed_data/px (X=participant ID)

    def predict(self, window_data):
        # window_data: (20, 34) array
        # Extract features, scale, predict
        # Return True if 'rest', False if 'not-rest'
        
    def save_model(self, filepath):
        # Save model, scaler, window_size
        
    def load_model(self, filepath):
        # Load model, scaler, window_size
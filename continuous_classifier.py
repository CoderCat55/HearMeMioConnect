import numpy as np
import time
from collections import deque
import pickle
from sklearn import svm
from sklearn.preprocessing import StandardScaler

class ContinuousClassifier:
    def __init__(self, emg_queue, classify_flag):
        self.emg_queue = emg_queue
        self.classify_flag = classify_flag
        
        # Sliding window for real-time classification
        self.window_size = 150  # 0.75 seconds at 200Hz
        self.emg_window = deque(maxlen=self.window_size)
        
        # Load trained model (you'll train this first)
        self.model = None
        self.scaler = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained SVM model"""
        try:
            with open('gesture_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
            print("Model loaded successfully")
        except:
            print("No trained model found. Please train first.")
            self.model = None
    
    def run(self):
        """Main classification loop"""
        print("Continuous Classifier started...")
        
        while True:
            try:
                # Get data from queue
                if not self.emg_queue.empty():
                    data_packet = self.emg_queue.get_nowait()
                    
                    if data_packet['type'] == 'emg':
                        self.emg_window.append(data_packet)
                        
                        # Classify if we have enough data and classification is active
                        if (self.classify_flag.value and 
                            len(self.emg_window) >= self.window_size and 
                            self.model is not None):
                            
                            prediction = self.classify_current_gesture()
                            if prediction:
                                print(f"Predicted gesture: {prediction}")
                                
            except:
                time.sleep(0.001)
    
    def classify_current_gesture(self):
        """Classify gesture from current window"""
        # Extract features from window
        features = self.extract_features(self.emg_window)
        
        if features is not None:
            # Scale features and predict
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            confidence = np.max(self.model.predict_proba(features_scaled))
            
            return {
                'gesture': prediction,
                'confidence': float(confidence),
                'timestamp': time.time()
            }
        return None
    
    def extract_features(self, window):
        """Extract features from EMG window (simplified version)"""
        if len(window) < self.window_size:
            return None
            
        # Convert to numpy array
        emg_data = np.array([item['data'] for item in list(window)[-self.window_size:]])
        
        features = []
        
        # Extract time-domain features for each channel
        for channel in range(8):  # 8 EMG channels
            channel_data = emg_data[:, channel]
            
            # Mean Absolute Value
            mav = np.mean(np.abs(channel_data))
            # Root Mean Square
            rms = np.sqrt(np.mean(channel_data**2))
            # Waveform Length
            wl = np.sum(np.abs(np.diff(channel_data)))
            # Zero Crossings
            zc = np.sum(np.diff(np.sign(channel_data)) != 0)
            
            features.extend([mav, rms, wl, zc])
        
        return np.array(features)
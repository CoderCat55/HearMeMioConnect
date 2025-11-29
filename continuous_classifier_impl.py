"""
Continuous Classifier Process
Performs real-time gesture recognition using trained SVM model
"""

import numpy as np
from collections import deque
import time
import queue
import os

WINDOW_SIZE = 150  # Samples (0.75 seconds at 200Hz)
OVERLAP = 100      # Samples
SHIFT = WINDOW_SIZE - OVERLAP  # 50 samples = 0.25 seconds


def continuous_classifier_process(emg_queue, result_queue, classify_flag, shutdown_event):
    """
    Main function for Continuous Classifier process
    
    Args:
        emg_queue: Queue receiving EMG+IMU data packets
        result_queue: Queue to put classification results
        classify_flag: Shared boolean - True when classifying
        shutdown_event: Event to signal shutdown
    """
    print("🟢 Continuous Classifier: Started")
    
    # Load model
    model, scaler = load_model()
    
    if model is None:
        print("⚠️  Continuous Classifier: No model loaded, classification disabled")
    
    # Buffers for each armband
    buffers = {
        0: deque(maxlen=WINDOW_SIZE),
        1: deque(maxlen=WINDOW_SIZE)
    }
    
    last_classification_time = 0
    classification_interval = 0.25  # 250ms between predictions
    
    while not shutdown_event.is_set():
        try:
            data = emg_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️  Classifier: Error reading queue: {e}")
            continue
        
        armband_id = data['armband_id']
        
        # Combine EMG + IMU (same as logger)
        try:
            combined = (
                data['emg'] + 
                data['imu']['accel'] + 
                data['imu']['gyro'] + 
                data['imu']['orientation'][1:]
            )
        except Exception as e:
            print(f"⚠️  Classifier: Error combining data: {e}")
            continue
        
        if len(combined) != 17:
            continue
        
        # Add to buffer
        buffers[armband_id].append(combined)
        
        # Check if should classify
        current_time = time.time()
        
        if classify_flag.value and \
           model is not None and \
           len(buffers[0]) == WINDOW_SIZE and \
           len(buffers[1]) == WINDOW_SIZE and \
           (current_time - last_classification_time) >= classification_interval:
            
            # Create window
            window = np.zeros((WINDOW_SIZE, 2, 17))
            window[:, 0, :] = np.array(buffers[0])
            window[:, 1, :] = np.array(buffers[1])
            
            # Extract features
            features = extract_features(window)
            
            # Predict
            try:
                features_scaled = scaler.transform([features])
                gesture = model.predict(features_scaled)[0]
                proba = model.predict_proba(features_scaled)[0]
                confidence = np.max(proba)
                
                # Create result
                result = {
                    'gesture': str(gesture),
                    'confidence': float(confidence),
                    'timestamp': current_time,
                    'probabilities': {
                        str(class_name): float(prob) 
                        for class_name, prob in zip(model.classes_, proba)
                    }
                }
                
                # Put in result queue (non-blocking, replace old if full)
                try:
                    result_queue.put(result, block=False)
                except queue.Full:
                    # Clear one old result and try again
                    try:
                        result_queue.get_nowait()
                        result_queue.put(result, block=False)
                    except:
                        pass
                
                print(f"🎯 Classified: {gesture} (confidence: {confidence:.2f})")
                
            except Exception as e:
                print(f"⚠️  Classifier: Prediction error: {e}")
            
            last_classification_time = current_time
            
            # Shift window
            for _ in range(SHIFT):
                buffers[0].popleft()
                buffers[1].popleft()
    
    print("🔴 Continuous Classifier: Stopped")


def load_model():
    """Load trained SVM model and scaler"""
    try:
        import joblib
        model = joblib.load('gesture_model.pkl')
        scaler = joblib.load('gesture_scaler.pkl')
        print("✅ Continuous Classifier: Model loaded successfully")
        print(f"   Classes: {model.classes_}")
        return model, scaler
    except FileNotFoundError:
        print("⚠️  Continuous Classifier: Model files not found")
        return None, None
    except Exception as e:
        print(f"⚠️  Continuous Classifier: Error loading model: {e}")
        return None, None


def extract_features(window):
    """
    Extract features from window for real-time classification
    
    Args:
        window: numpy array, shape (150, 2, 17)
                - 150 samples
                - 2 armbands
                - 17 channels (8 EMG + 9 IMU)
    
    Returns:
        1D feature vector (110 features)
    """
    features = []
    
    for armband in range(2):
        # ============================================================
        # EMG FEATURES (channels 0-7)
        # ============================================================
        for ch in range(8):
            emg_data = window[:, armband, ch]
            
            # Time-domain EMG features
            mav = np.mean(np.abs(emg_data))                      # Mean Absolute Value
            rms = np.sqrt(np.mean(emg_data**2))                  # Root Mean Square
            wl = np.sum(np.abs(np.diff(emg_data)))               # Waveform Length
            zc = np.sum(np.diff(np.sign(emg_data)) != 0)         # Zero Crossings
            ssc = np.sum(np.diff(np.sign(np.diff(emg_data))) != 0)  # Slope Sign Changes
            
            features.extend([mav, rms, wl, zc, ssc])
            # 8 channels × 5 features = 40 features per armband
        
        # ============================================================
        # IMU FEATURES
        # ============================================================
        
        # Accelerometer (channels 8, 9, 10) - x, y, z
        accel_data = window[:, armband, 8:11]
        accel_mean = np.mean(accel_data, axis=0)  # Mean of x, y, z
        accel_std = np.std(accel_data, axis=0)    # Std of x, y, z
        features.extend(accel_mean)  # 3 features
        features.extend(accel_std)   # 3 features
        
        # Gyroscope (channels 11, 12, 13) - x, y, z
        gyro_data = window[:, armband, 11:14]
        gyro_mean = np.mean(gyro_data, axis=0)
        gyro_std = np.std(gyro_data, axis=0)
        features.extend(gyro_mean)   # 3 features
        features.extend(gyro_std)    # 3 features
        
        # Orientation (channels 14, 15, 16) - quaternion x, y, z
        orient_data = window[:, armband, 14:17]
        orient_mean = np.mean(orient_data, axis=0)
        features.extend(orient_mean)  # 3 features
        
        # Total IMU features per armband: 3 + 3 + 3 + 3 + 3 = 15
    
    # Total features: 2 armbands × (40 EMG + 15 IMU) = 110 features
    return np.array(features)

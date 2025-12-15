#!/usr/bin/env python3
"""
Raspberry Pi 5 Web Server - ESP32 Functionality Port
Provides gesture recognition endpoints for mobile app access
"""

from flask import Flask, jsonify, request
import threading
import time
import math

app = Flask(__name__)

# Configuration
SSID = "duybeni"
PASSWORD = "123456789"
PORT = 5002

# Sensor data structure (will be populated from your shared memory)
sensor_data = {
    'ax1': 0.0, 'ay1': 0.0, 'az1': 0.0,  # Accelerometer 1
    'ax2': 0.0, 'ay2': 0.0, 'az2': 0.0,  # Accelerometer 2
    'emg1': 0.0,  # EM.G sensor 1
    'emg2': 0.0,  # EMG sensor 2
    'cr': 0,      # Classification result
    'calword': 0,
    'connections': 0
}

# Calibration data
MAX_SAMPLES = 20
NUM_WORDS = 10
NUM_FEATURES = 8  # 2 EMG + 6 Accel

calibration_index = [0] * NUM_WORDS
calibration_data = [[[0.0] * NUM_FEATURES for _ in range(MAX_SAMPLES)] for _ in range(NUM_WORDS)]

# EMG processing buffers
class EMGProcessor:
    def __init__(self, buffer_size=20):
        self.buffer = [0.0] * buffer_size
        self.index = 0
        self.sum_squares = 0.0
        self.buffer_size = buffer_size
    
    def process(self, value):
        """RMS filter for EMG"""
        self.sum_squares -= self.buffer[self.index] ** 2
        self.buffer[self.index] = value
        self.sum_squares += value ** 2
        self.index = (self.index + 1) % self.buffer_size
        return math.sqrt(self.sum_squares / self.buffer_size)

emg1_processor = EMGProcessor()
emg2_processor = EMGProcessor()


def classify_gesture():
    """
    K-NN gesture classification with distance-weighted voting
    Returns: gesture number (1-10) or 0 if no confident match
    """
    print("classify gesture fonksiyonun çağırınız")
    return 0


def save_calibration_sample(word_index):
    """Save current sensor data as calibration sample"""
    if word_index < 0 or word_index >= NUM_WORDS:
        return False
    
    # FIFO - shift if buffer full
    if calibration_index[word_index] >= MAX_SAMPLES:
        for i in range(MAX_SAMPLES - 1):
            calibration_data[word_index][i] = calibration_data[word_index][i + 1][:]
        calibration_index[word_index] = MAX_SAMPLES - 1
    
    # Store current data
    idx = calibration_index[word_index]
    calibration_data[word_index][idx][0] = emg1_processor.process(sensor_data['emg1'])
    calibration_data[word_index][idx][1] = emg2_processor.process(sensor_data['emg2'])
    calibration_data[word_index][idx][2] = sensor_data['ax1']
    calibration_data[word_index][idx][3] = sensor_data['ay1']
    calibration_data[word_index][idx][4] = sensor_data['az1']
    calibration_data[word_index][idx][5] = sensor_data['ax2']
    calibration_data[word_index][idx][6] = sensor_data['ay2']
    calibration_data[word_index][idx][7] = sensor_data['az2']
    
    calibration_index[word_index] += 1
    print(f"Saved sample {calibration_index[word_index]} for word {word_index + 1}")
    return True


def clear_calibration_data(word_index):
    """Clear calibration data for a word"""
    if word_index < 0 or word_index >= NUM_WORDS:
        return False
    
    calibration_index[word_index] = 0
    print(f"Cleared all data for word {word_index + 1}")
    return True


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Root endpoint"""
    sensor_data['connections'] += 1
    return jsonify({"status": "ok", "message": "Raspberry Pi Gesture Recognition Server"})


@app.route('/data')
def get_data():
    """Get current sensor data"""
    return jsonify(sensor_data)


@app.route('/classify')
def classify():
    """Perform gesture classification"""
    sensor_data['cr'] = classify_gesture()
    print(f"Classification requested - Result: {sensor_data['cr']}")
    return jsonify(sensor_data)


@app.route('/setcw')
def set_calibration_word():
    """Save calibration sample for a word"""
    value = request.args.get('value', type=int)
    print("you called setcw functiion")
    if value is None:
        return "Missing value parameter", 400
    
    if value < 1 or value > NUM_WORDS:
        return f"Invalid word index (1-{NUM_WORDS})", 400
    
    if save_calibration_sample(value - 1):
        return f"Saved sample for word {value}", 200
    else:
        return "Error saving sample", 500


@app.route('/deletecw')
def delete_calibration_word():
    """Clear calibration data for a word"""
    value = request.args.get('value', type=int)
    print("you called deletecw functiion")
    
    if value is None:
        return "Missing value parameter", 400
    
    if value < 1 or value > NUM_WORDS:
        return f"Invalid word index (1-{NUM_WORDS})", 400
    
    if clear_calibration_data(value - 1):
        return f"Cleared calibration for word {value}", 200
    else:
        return "Error clearing calibration", 500


# ==================== Helper Functions ====================

def update_sensor_data_from_shared_memory():
    """
    PLACEHOLDER: Implement this function to read from your shared memory
     """
    print("you called update snespr data from shared memory")
    pass


def sensor_update_loop():
    """Background thread to continuously update sensor data"""
    while True:
        try:
            update_sensor_data_from_shared_memory()
            time.sleep(0.01)  # 100Hz update rate
        except Exception as e:
            print(f"Sensor update error: {e}")
            time.sleep(0.1)


# ==================== Main ====================

if __name__ == '__main__':
    print("=" * 50)
    print("Raspberry Pi Gesture Recognition Server")
    print("=" * 50)
    print(f"Access via: http://hmrasphost.local:{PORT}/")
    print(f"Or via AP IP: http://192.168.4.1:{PORT}/")
    print(f"WiFi AP SSID: {SSID}")
    print(f"WiFi AP Password: {PASSWORD}")
    print("=" * 50)
    
    # Start sensor update thread (optional - uncomment if needed)
    # sensor_thread = threading.Thread(target=sensor_update_loop, daemon=True)
    # sensor_thread.start()
    
    # Start Flask server
    try:
        app.run(host='0.0.0.0', port=PORT, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
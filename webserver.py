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

@app.route('/')
def index():
    """"""
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
#!/usr/bin/env python3
"""
Raspberry Pi 5 Web Server - ESP32 Functionality Port
Provides gesture recognition endpoints for mobile app access
"""

from flask import Flask, jsonify, request
import threading
import time
import math
from main import *

app = Flask(__name__)

# Configuration
SSID="hearme"
PASSWORD="123456789"
PORT = 5002

# Sensor data structure (will be populated from your shared memory)
sensor_data = {
    'ituemg0': 0, 'ituemg1': 0,'ituemg2': 0,'ituemg3': 0,'ituemg4': 0,'ituemg5': 0,'ituemg6': 0,'ituemg7': 0,
    'ituax': 0,'ituay': 0,'ituaz': 0,'itugx': 0,'itugy': 0,'itugz': 0,'ituroll': 0,'ituyaw': 0,'itupitch': 0,
    'maremg0': 0,'maremg1': 0,'maremg2': 0,'maremg3': 0,'maremg4': 0,'maremg5': 0,'maremg6': 0,'maremg7': 0,
    'marax': 0,'maray': 0,'maraz': 0,'margx': 0,'margy': 0,'margz': 0,'marroll': 0,'maryaw': 0,'marpitch': 0,
    'cr': 0,      # Classification result
    'calword': 0,
    'connections': 0
}

def clear_calibration_data(word_index):
    """Clear calibration data for a word"""
    return True


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Root endpoint"""
    sensor_data['connections'] += 1
    return jsonify({"status": "ok", "message": "Raspberry Pi Gesture Recognition Server"})

@app.route('/data')
def data():
    """Get current sensor data from shared memory and update """
    return jsonify(sensor_data)


@app.route('/classify')
def classify():
    """Perform gesture classification"""
    Classify(stream_buffer, stream_index, classifier)
    
    #return jsonify(sensor_data)


@app.route('/setcw')
def setcw():
    """Save calibration sample for a word"""
    value = request.args.get('value', type=int)
    Calibrate(value, calib_buffer, calib_index, recording_flag, recording_gesture, classifier)

@app.route('/train')
def train():
    """Train the model"""
    Train(classifier)

@app.route('/connect')
def connect():
    """Data aqusition process should start when this is fetched"""
    

# ==================== Main ====================

if __name__ == '__main__':
    print("=" * 50)
    print("Raspberry Pi Gesture Recognition Server")
    print("=" * 50)
    print(f"Access via: http://hmrasphost.local:{PORT}/")
    print(f"Or via AP IP: http://192.168.4.1:{PORT}/"+"this one works") 
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
#!/usr/bin/env python3
"""
Raspberry Pi 5 Web Server - Gesture Recognition API
Provides gesture recognition endpoints for mobile app access
"""

from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

# Configuration
SSID = "hearme"
PASSWORD = "123456789"
PORT = 5002

# Global system reference (will be injected from main.py)
_system = None

def inject_system(system):
    """Inject the GestureSystem instance from main.py"""
    global _system
    _system = system
    print("✓ GestureSystem injected into webserver")


# Sensor data structure (for compatibility with existing mobile app)
sensor_data = {
    'ituemg0': 0, 'ituemg1': 0, 'ituemg2': 0, 'ituemg3': 0, 
    'ituemg4': 0, 'ituemg5': 0, 'ituemg6': 0, 'ituemg7': 0,
    'ituax': 0, 'ituay': 0, 'ituaz': 0, 'itugx': 0, 'itugy': 0, 
    'itugz': 0, 'ituroll': 0, 'ituyaw': 0, 'itupitch': 0,
    'maremg0': 0, 'maremg1': 0, 'maremg2': 0, 'maremg3': 0, 
    'maremg4': 0, 'maremg5': 0, 'maremg6': 0, 'maremg7': 0,
    'marax': 0, 'maray': 0, 'maraz': 0, 'margx': 0, 'margy': 0, 
    'margz': 0, 'marroll': 0, 'maryaw': 0, 'marpitch': 0,
    #'cr': 0,      # Classification result
    'calword': 0,
    'connections': 0 
}


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Root endpoint"""
    sensor_data['connections'] += 1
    return jsonify({
        "status": "ok", 
        "message": "Raspberry Pi Gesture Recognition Server",
        "data_acquisition_running": _system.is_data_acquisition_running() if _system else False
    })


@app.route('/data')
def data():
    """Get current sensor data from shared memory"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    if not _system.is_data_acquisition_running():
        return jsonify({
            "status": "error",
            "message": "Data acquisition not running. Call /connect first."
        }), 400
    
    # Read last sample from circular buffer (NO modification to shared memory)
    current_idx = _system.stream_index.value
    if current_idx == 0:
        return jsonify({
            "status": "no_data",
            "message": "No data available yet. Wait for sensors to start streaming."
        })
    
    # Get the most recent sample (use modulo for circular buffer)
    from main import STREAM_BUFFER_SIZE
    buffer_idx = (current_idx - 1) % STREAM_BUFFER_SIZE
    latest_sample = _system.stream_buffer[buffer_idx].copy()  # .copy() to avoid shared memory issues
    
    # Parse into mobile app format (alphabetically: MyoITU first, MyoMarmara second)
    sensor_data = {
        # First Myo (index 0-16)
        'ituemg0': float(latest_sample[0]), 'ituemg1': float(latest_sample[1]),
        'ituemg2': float(latest_sample[2]), 'ituemg3': float(latest_sample[3]),
        'ituemg4': float(latest_sample[4]), 'ituemg5': float(latest_sample[5]),
        'ituemg6': float(latest_sample[6]), 'ituemg7': float(latest_sample[7]),
        'ituroll': float(latest_sample[8]), 'itupitch': float(latest_sample[9]),
        'ituyaw': float(latest_sample[10]),
        'ituax': float(latest_sample[11]), 'ituay': float(latest_sample[12]),
        'ituaz': float(latest_sample[13]),
        'itugx': float(latest_sample[14]), 'itugy': float(latest_sample[15]),
        'itugz': float(latest_sample[16]),
        
        # Second Myo (index 17-33)
        'maremg0': float(latest_sample[17]), 'maremg1': float(latest_sample[18]),
        'maremg2': float(latest_sample[19]), 'maremg3': float(latest_sample[20]),
        'maremg4': float(latest_sample[21]), 'maremg5': float(latest_sample[22]),
        'maremg6': float(latest_sample[23]), 'maremg7': float(latest_sample[24]),
        'marroll': float(latest_sample[25]), 'marpitch': float(latest_sample[26]),
        'maryaw': float(latest_sample[27]),
        'marax': float(latest_sample[28]), 'maray': float(latest_sample[29]),
        'maraz': float(latest_sample[30]),
        'margx': float(latest_sample[31]), 'margy': float(latest_sample[32]),
        'margz': float(latest_sample[33]),
        
        'calword': 0,
        'connections': 0 #add if connections are available with myo armbands 
    }
    
    return jsonify(sensor_data)


@app.route('/connect')
def connect():
    """Start data acquisition process when mobile app connects"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    if _system.is_data_acquisition_running():
        return jsonify({
            "status": "already_connected",
            "message": "Data acquisition already running"
        })
    
    success = _system.start_data_acquisition()
    
    if success:
        return jsonify({
            "status": "connecting",
            "message": "Data acquisition started successfully"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Failed to start data acquisition"
        }), 500


@app.route('/disconnect')
def disconnect():
    """Stop data acquisition process"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    _system.stop_data_acquisition()
    
    return jsonify({
        "status": "disconnected",
        "message": "Data acquisition stopped"
    })


@app.route('/classify')
def classify():
    """Perform gesture classification"""
    """burada neden sensor datayı gönderiyoruz?????? cr yi göndermek için"""
    """aslında sensor datada cr ye ihtiyacımız yok direkt olarak cryi gönderebiliriz muhtemelen"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    if not _system.is_data_acquisition_running():
        return jsonify({
            "status": "error",
            "message": "Data acquisition not running. Call /connect first."
        }), 400
    
    cr = _system.classify()
    
    if cr is None:
        return jsonify({
            "status": "error",
            "message": "Classification failed"
        }), 500
    return jsonify({
        "status": "success",
        "message": cr,
    })


@app.route('/setcw')
def setcw():
    """Save calibration sample for a gesture"""
    """şimdi calibrasyon sayfasında verileri göstermek için /data kullanıyoruz belli bir intervalde çağırarak
    aynı şekilde calibrationu da belli bir intervalde çağırarak kullanıyorduk  bunu artık bir zaman serisinikaydetmek için kullanacağız
    yani kalibrasyon her çağrıldığında aslında 3snlik bir veri kaydedicek calibrastionu çağırmadan araya 5snlik bekleme koyabiiliriz
    ? bu sistem 1 defa calibrate endpointine gittiğinde ne kadar data kaydediyor Calibrationduration kadar mı ? evet"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    # Get gesture name/number from query params
    value = request.args.get('value', type=int)
    gesture_name = request.args.get('name', type=str)
    
    # Support both numeric IDs and string names
    if gesture_name is None and value is not None:
        gesture_name = f"gesture_{value}"
    elif gesture_name is None:
        return jsonify({
            "status": "error",
            "message": "Missing 'value' or 'name' parameter"
        }), 400
    
    if not _system.is_data_acquisition_running():
        return jsonify({
            "status": "error",
            "message": "Data acquisition not running. Call /connect first."
        }), 400
    
    success = _system.calibrate(gesture_name)
    
    if success:
        return jsonify({
            "status": "success",
            "message": f"Calibration sample saved for '{gesture_name}'"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Calibration failed"
        }), 500


@app.route('/train')
def train():
    """Train the model"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    success = _system.train()
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Model trained successfully"
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Training failed. Make sure you have calibration data."
        }), 500


@app.route('/status')
def status():
    """Get system status"""
    if _system is None:
        return jsonify({
            "status": "error",
            "message": "System not initialized"
        }), 500
    
    return jsonify({
        "status":"success",
        "message": {
            "data_acquisition_running": _system.is_data_acquisition_running(),
            "model_trained": _system.classifier.model is not None,
            "available_gestures": _system.classifier.gesture_labels,
            "calibration_samples": {
                gesture: len(samples) 
                for gesture, samples in _system.classifier.calibration_data.items()
            }
        }
    })


# ==================== Main ====================

if __name__ == '__main__':
    print("=" * 50)
    print("ERROR: Do not run webserver.py directly!")
    print("Please run main.py instead.")
    print("=" * 50)
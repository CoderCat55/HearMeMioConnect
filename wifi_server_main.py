"""
WiFi Server - Main Orchestrator
Launches all processes and provides HTTP API for mobile app
"""

from flask import Flask, request, jsonify
from multiprocessing import Process, Queue, Value, Array, Event
import signal
import sys
import time
import os
import glob
import numpy as np
import traceback

# Import your other modules
from myo_handler_template import myo_handler_process
from numpy_logger_impl import numpy_logger_process
from continuous_classifier_impl import continuous_classifier_process

app = Flask(__name__)

# ============================================================================
# GLOBAL SHARED MEMORY OBJECTS (Created on startup)
# ============================================================================
emg_queue_logger = None
emg_queue_classifier = None
result_queue = None
recording_flag = None
classify_flag = None
current_label = None
system_shutdown = None

# Process references
processes = {}


# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'recording': bool(recording_flag.value),
        'classifying': bool(classify_flag.value),
        'model_trained': os.path.exists('gesture_model.pkl'),
        'logger_queue_size': emg_queue_logger.qsize(),
        'classifier_queue_size': emg_queue_classifier.qsize(),
        'result_queue_size': result_queue.qsize(),
        'processes_alive': {
            name: proc.is_alive() for name, proc in processes.items()
        }
    })


@app.route('/record', methods=['POST'])
def start_recording():
    """Start recording training data
    
    Expected JSON: {"gesture": "fist"}
    """
    try:
        gesture_label = request.json.get('gesture', 'unknown')
        
        if not gesture_label or gesture_label == 'unknown':
            return jsonify({'error': 'Gesture label required'}), 400
        
        # Set shared memory
        label_bytes = gesture_label.encode('utf-8')[:49]  # Max 49 chars + null
        current_label.value = label_bytes + b'\x00' * (50 - len(label_bytes))
        recording_flag.value = True
        
        return jsonify({
            'status': 'recording',
            'gesture': gesture_label
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop_record', methods=['POST'])
def stop_recording():
    """Stop recording training data"""
    recording_flag.value = False
    
    # Give logger time to save file
    time.sleep(0.5)
    
    return jsonify({'status': 'stopped'})


@app.route('/classify/start', methods=['POST'])
def start_classification():
    """Start continuous classification"""
    if not os.path.exists('gesture_model.pkl'):
        return jsonify({'error': 'Model not trained yet'}), 400
    
    classify_flag.value = True
    
    # Clear old results
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except:
            break
    
    return jsonify({'status': 'classifying'})


@app.route('/classify/results', methods=['GET'])
def get_classification_results():
    """Get latest classification result
    
    This endpoint should be polled by mobile app every 300ms
    """
    try:
        # Non-blocking get with short timeout
        result = result_queue.get(timeout=0.1)
        return jsonify(result)
    
    except:
        # No new results available
        return jsonify({'status': 'waiting'}), 204


@app.route('/classify/stop', methods=['POST'])
def stop_classification():
    """Stop continuous classification"""
    classify_flag.value = False
    return jsonify({'status': 'stopped'})


@app.route('/train', methods=['POST'])
def train_model():
    """Train SVM model from all collected .npz files"""
    try:
        import joblib
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        print("Starting model training...")
        
        # Load all training files
        npz_files = glob.glob('training_*.npz')
        
        if len(npz_files) == 0:
            return jsonify({'error': 'No training data found'}), 404
        
        print(f"Found {len(npz_files)} training files")
        
        all_windows = []
        all_labels = []
        
        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            all_windows.append(data['windows'])
            all_labels.append(data['labels'])
            print(f"  Loaded {npz_file}: {len(data['labels'])} examples")
        
        # Combine all
        X_windows = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        print(f"Total examples: {len(y)}")
        print(f"Gestures: {set(y)}")
        
        # Extract features from all windows
        print("Extracting features...")
        X_features = np.array([extract_features_for_training(window) for window in X_windows])
        
        print(f"Feature shape: {X_features.shape}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM
        print("Training SVM...")
        model = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        
        print(f"Train accuracy: {train_acc:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
        
        # Save
        joblib.dump(model, 'gesture_model.pkl')
        joblib.dump(scaler, 'gesture_scaler.pkl')
        
        print("Model saved!")
        
        return jsonify({
            'status': 'trained',
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'gestures': list(set(y)),
            'num_examples': len(X_features),
            'num_features': X_features.shape[1],
            'training_files_used': len(npz_files)
        })
    
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/data/list', methods=['GET'])
def list_training_data():
    """List all collected training data files"""
    npz_files = glob.glob('training_*.npz')
    
    file_info = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        metadata = data['metadata'].item() if 'metadata' in data else {}
        
        file_info.append({
            'filename': npz_file,
            'gesture': metadata.get('gesture', 'unknown'),
            'num_examples': len(data['labels']) if 'labels' in data else 0,
            'recording_date': metadata.get('recording_date', 'unknown')
        })
    
    return jsonify({
        'total_files': len(npz_files),
        'files': file_info
    })


@app.route('/data/delete', methods=['POST'])
def delete_training_data():
    """Delete specific training file
    
    Expected JSON: {"filename": "training_fist_20250115.npz"}
    """
    filename = request.json.get('filename')
    
    if not filename or not filename.startswith('training_'):
        return jsonify({'error': 'Invalid filename'}), 400
    
    try:
        os.remove(filename)
        return jsonify({'status': 'deleted', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown_system():
    """Gracefully shutdown all processes"""
    print("Shutdown requested via API")
    system_shutdown.set()
    
    # Give processes time to cleanup
    time.sleep(1)
    
    return jsonify({'status': 'shutting_down'})


# ============================================================================
# FEATURE EXTRACTION (Same as in classifier, but imported here for training)
# ============================================================================

def extract_features_for_training(window):
    """
    Extract features from window for training
    window shape: (150, 2, 17)
    - 150 samples
    - 2 armbands
    - 17 channels (8 EMG + 9 IMU)
    
    Returns: 1D feature vector
    """
    features = []
    
    for armband in range(2):
        # EMG features (channels 0-7)
        for ch in range(8):
            emg_data = window[:, armband, ch]
            
            # Time-domain EMG features
            mav = np.mean(np.abs(emg_data))
            rms = np.sqrt(np.mean(emg_data**2))
            wl = np.sum(np.abs(np.diff(emg_data)))
            zc = np.sum(np.diff(np.sign(emg_data)) != 0)
            ssc = np.sum(np.diff(np.sign(np.diff(emg_data))) != 0)
            
            features.extend([mav, rms, wl, zc, ssc])
        
        # IMU features (channels 8-16)
        # Accelerometer (8, 9, 10)
        accel_data = window[:, armband, 8:11]
        accel_mean = np.mean(accel_data, axis=0)
        accel_std = np.std(accel_data, axis=0)
        features.extend(accel_mean)
        features.extend(accel_std)
        
        # Gyroscope (11, 12, 13)
        gyro_data = window[:, armband, 11:14]
        gyro_mean = np.mean(gyro_data, axis=0)
        gyro_std = np.std(gyro_data, axis=0)
        features.extend(gyro_mean)
        features.extend(gyro_std)
        
        # Orientation (14, 15, 16)
        orient_data = window[:, armband, 14:17]
        orient_mean = np.mean(orient_data, axis=0)
        features.extend(orient_mean)
    
    return np.array(features)


# ============================================================================
# PROCESS LAUNCHER
# ============================================================================

def launch_all_processes():
    """Launch all background processes"""
    global emg_queue_logger, emg_queue_classifier, result_queue
    global recording_flag, classify_flag, current_label, system_shutdown
    global processes
    
    print("="*60)
    print("INITIALIZING GESTURE RECOGNITION SYSTEM")
    print("="*60)
    
    # Create shared memory objects
    print("\n[1/5] Creating shared memory objects...")
    emg_queue_logger = Queue(maxsize=1000)
    emg_queue_classifier = Queue(maxsize=1000)
    result_queue = Queue(maxsize=20)
    recording_flag = Value('b', False)
    classify_flag = Value('b', False)
    current_label = Array('c', b' ' * 50)
    system_shutdown = Event()
    print("  ✓ Shared memory created")
    
    # Launch Process 1: Myo Handler
    print("\n[2/5] Launching Myo Handler...")
    p1 = Process(
        target=myo_handler_process,
        args=(emg_queue_logger, emg_queue_classifier, system_shutdown),
        name="MyoHandler"
    )
    p1.start()
    processes['myo_handler'] = p1
    print("  ✓ Myo Handler started (PID: {})".format(p1.pid))
    
    # Launch Process 2: Numpy Logger
    print("\n[3/5] Launching Numpy Logger...")
    p2 = Process(
        target=numpy_logger_process,
        args=(emg_queue_logger, recording_flag, current_label, system_shutdown),
        name="NumpyLogger"
    )
    p2.start()
    processes['numpy_logger'] = p2
    print("  ✓ Numpy Logger started (PID: {})".format(p2.pid))
    
    # Launch Process 3: Continuous Classifier
    print("\n[4/5] Launching Continuous Classifier...")
    p3 = Process(
        target=continuous_classifier_process,
        args=(emg_queue_classifier, result_queue, classify_flag, system_shutdown),
        name="Classifier"
    )
    p3.start()
    processes['classifier'] = p3
    print("  ✓ Classifier started (PID: {})".format(p3.pid))
    
    print("\n[5/5] All processes launched successfully!")
    print("="*60)
    print("SYSTEM READY")
    print("="*60)
    print(f"\nWiFi Server will start on http://0.0.0.0:5000")
    print("Mobile app can connect to: http://<raspberry-pi-ip>:5000")
    print("\nPress Ctrl+C to shutdown gracefully\n")


def cleanup_processes():
    """Gracefully shutdown all processes"""
    print("\n" + "="*60)
    print("SHUTTING DOWN SYSTEM")
    print("="*60)
    
    system_shutdown.set()
    
    for name, proc in processes.items():
        print(f"Waiting for {name} to stop...")
        proc.join(timeout=3)
        if proc.is_alive():
            print(f"  Force terminating {name}...")
            proc.terminate()
            proc.join(timeout=1)
        print(f"  ✓ {name} stopped")
    
    print("\n✓ All processes stopped")
    print("="*60)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    cleanup_processes()
    sys.exit(0)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Launch all background processes
    launch_all_processes()
    
    # Run Flask server (this blocks until shutdown)
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_processes()

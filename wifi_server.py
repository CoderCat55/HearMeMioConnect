from flask import Flask, request, jsonify
from multiprocessing import Process, Queue, Value, Array, Event
import signal
import sys
import time
import numpy as np
import pickle
from src.myodriver import MyoDriver
from src.config import Config

app = Flask(__name__)

class GestureRecognitionSystem:
    def __init__(self):
        # Shared memory for process communication
        self.emg_queue_logger = Queue(maxsize=1000)
        self.emg_queue_classifier = Queue(maxsize=1000)
        self.recording_flag = Value('b', False)  # Shared boolean
        self.classify_flag = Value('b', False)   # Shared boolean
        self.current_label = Array('c', b' ' * 50)  # Shared string
        self.system_shutdown = Event()
        
        # Processes
        self.myo_driver_process = None
        self.numpy_logger_process = None
        self.classifier_process = None
        
        # System state
        self.is_running = False
        
    def start_system(self):
        """Start all processes"""
        if self.is_running:
            return {"status": "already_running"}
            
        # Start Myo Driver process
        self.myo_driver_process = Process(
            target=self._run_myo_driver,
            args=(self.emg_queue_logger, self.emg_queue_classifier)
        )
        self.myo_driver_process.start()
        
        # Start Numpy Logger process
        self.numpy_logger_process = Process(
            target=self._run_numpy_logger,
            args=(self.emg_queue_logger, self.recording_flag, self.current_label)
        )
        self.numpy_logger_process.start()
        
        # Start Classifier process
        self.classifier_process = Process(
            target=self._run_classifier,
            args=(self.emg_queue_classifier, self.classify_flag)
        )
        self.classifier_process.start()
        
        self.is_running = True
        return {"status": "started"}
    
    def _run_myo_driver(self, emg_queue_logger, emg_queue_classifier):
        """Run Myo driver in a separate process"""
        config = Config()
        config.MYO_AMOUNT = 2
        config.PRINT_EMG = False  # We'll handle printing in main process
        config.PRINT_IMU = False
        
        myo_driver = MyoDriver(config, emg_queue_logger, emg_queue_classifier)
        myo_driver.run()
        
        # Keep process alive
        while not self.system_shutdown.is_set():
            myo_driver.receive()
            time.sleep(0.001)
    
    def _run_numpy_logger(self, emg_queue, recording_flag, current_label):
        """Run numpy logger in separate process"""
        from numpy_logger import NumpyLogger
        logger = NumpyLogger(emg_queue, recording_flag, current_label)
        logger.run()
    
    def _run_classifier(self, emg_queue, classify_flag):
        """Run classifier in separate process"""
        from continuous_classifier import ContinuousClassifier
        classifier = ContinuousClassifier(emg_queue, classify_flag)
        classifier.run()
    
    def stop_system(self):
        """Stop all processes"""
        self.system_shutdown.set()
        
        if self.myo_driver_process:
            self.myo_driver_process.terminate()
        if self.numpy_logger_process:
            self.numpy_logger_process.terminate()
        if self.classifier_process:
            self.classifier_process.terminate()
            
        self.is_running = False
        return {"status": "stopped"}

# Global system instance
system = GestureRecognitionSystem()

# HTTP Routes
@app.route('/start_record', methods=['POST'])
def start_record():
    """Start recording training data"""
    gesture = request.json.get('gesture', 'unknown')
    
    system.recording_flag.value = True
    system.current_label.value = gesture.encode()[:49] + b'\0'
    
    return {"status": "recording_started", "gesture": gesture}

@app.route('/stop_record', methods=['POST'])
def stop_record():
    """Stop recording training data"""
    system.recording_flag.value = False
    return {"status": "recording_stopped"}

@app.route('/start_classify', methods=['POST'])
def start_classify():
    """Start continuous classification"""
    system.classify_flag.value = True
    return {"status": "classification_started"}

@app.route('/stop_classify', methods=['POST'])
def stop_classify():
    """Stop continuous classification"""
    system.classify_flag.value = False
    return {"status": "classification_stopped"}

@app.route('/train', methods=['POST'])
def train_model():
    """Train SVM model from collected data"""
    # This would trigger training on collected .npz files
    return {"status": "training_started"}

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    return {
        "running": system.is_running,
        "recording": bool(system.recording_flag.value),
        "classifying": bool(system.classify_flag.value),
        "current_gesture": system.current_label.value.decode().strip()
    }

if __name__ == "__main__":
    # Start the system
    print("Starting Gesture Recognition System...")
    system.start_system()
    
    # Start Flask server
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        system.stop_system()
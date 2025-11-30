"""handles multiprocessing
handles user commands
"""
# main.py
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, shared_memory
import numpy as np
from sckit import GestureClassifier
import sys
import time
from mioconnect.src.myodriver import MyoDriver
from mioconnect.src.config import Config

config = Config()
myo_driver = MyoDriver(config)

# Shared resources
shared_mem = None
recording_flag = mp.Value('i', 0)  # 0 = not recording, 1 = recording
recording_gesture = mp.Array('c', 50)  # Gesture name (50 char max)
result_queue = Queue()  # For passing classification results

def data_acquisition_process(shared_mem_name, recording_flag, recording_gesture):
    """Process 1: Continuously acquires data from Myo armbands"""
    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shared_mem_name)
    buffer = np.ndarray((1000, 34), dtype=np.float32, buffer=shm.buf)
    
    # Initialize MyoDriver
    config = Config()
    myo_driver = MyoDriver(config, shared_mem=buffer, 
                           recording_flag=recording_flag,
                           recording_gesture=recording_gesture)
    
    # Run forever
    while True:
        myo_driver.receive()



def Calibrate(gesture_name):
    """Called from main process when user wants to calibrate"""
    print(f"Recording calibration for '{gesture_name}' - 3 seconds...")
    
    # Set flag in shared memory
    recording_gesture.value = gesture_name.encode('utf-8')
    recording_flag.value = 1  # Start recording
    
    # Wait 3 seconds
    time.sleep(3.0)
    
    # Stop recording
    recording_flag.value = 0
    
    # Read recorded data from shared memory
    # (data_acquisition_process wrote it to a calibration buffer)
    recorded_data = get_calibration_buffer_from_shared_mem()
    
    # Add to classifier (runs in main process)
    classifier.add_calibration_sample(gesture_name, recorded_data)
    
    # Save to disk
    np.save(f'calibration_data/{gesture_name}_{timestamp}.npy', recorded_data)
    
    print(f"Calibration complete! Saved {len(recorded_data)} samples")

def Classify():
    """Called from main process when user wants to classify"""
    print("Classifying gesture...")
    
    # Read current data from shared memory (last 1 second = ~200 samples)
    current_data = get_recent_data_from_shared_mem(window_seconds=1.0)
    
    # Extract features
    features = extract_features(current_data)
    
    # Classify
    result = classifier.classify(features)
    print(f"Predicted gesture: {result}")
    return result

def Train():
    """Called from main process when user wants to train"""
    print("Training model...")
    classifier.train()
    classifier.save_model('model.pkl')
    print("Training complete!")

def Command():
    value = input("Enter your command majesty: ")
    match value:
        case "train":
            print("now will run train function")
            Train()
        case "classify":
            print("now will run classify function")
            Classify()
        case "calibrate":
            print("now will run calibrate function")
            gesture_name = input("Welche gesture möchten Sie calibrate?")
            Calibrate(gesture_name)
        case _:
            print("doğru düzgün komut yazsana biz böyle mi çalışacaz?")

if __name__ == "__main__":
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=1000*34*4)  # 1000 samples, 34 features, float32
    shared_buffer = np.ndarray((1000, 34), dtype=np.float32, buffer=shm.buf)
    
    # Start data acquisition process
    data_process = Process(target=data_acquisition_process, 
                          args=(shm.name, recording_flag, recording_gesture))
    data_process.start()
    
    # Initialize classifier in main process
    classifier = GestureClassifier()
    classifier.load_calibration_data()  # Load previous calibrations
    
    # Command loop
    while True:
        Command()  # Your existing command function
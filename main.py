"""handles multiprocessing and user commands"""
import multiprocessing as mp
from multiprocessing import Process, shared_memory
import numpy as np
import time
from rest_model import RestDetector
from gesture_model import GestureModel
import os

# Constants
STREAM_BUFFER_SIZE = 1000  # ~5 seconds at 200Hz
CALIBRATION_BUFFER_SIZE = 600  # 3 seconds at 200Hz

CALIBRATION_DURATION = 3  # seconds #no longer needed
CLASSIFICATION_DURATION = 3  # same! #no longer needed

CALIBRATION_STARTS = 5 
CLASSIFICATION_STARTS = 5 

def data_acquisition_process(stream_mem_name, calib_mem_name, stream_index, 
                            calib_index, recording_flag, recording_gesture):
    """Process 1: Continuously acquires data from Myo armbands"""
    from mioconnect.src.myodriver import MyoDriver
    from mioconnect.src.config import Config
    
    # Attach to shared memory
    shm_stream = shared_memory.SharedMemory(name=stream_mem_name)
    stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_stream.buf)
    
    shm_calib = shared_memory.SharedMemory(name=calib_mem_name)
    calib_buffer = np.ndarray((CALIBRATION_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_calib.buf)
    
    # Initialize MyoDriver with ALL parameters
    config = Config()
    myo_driver = MyoDriver(
        config, 
        stream_buffer=stream_buffer,
        stream_index=stream_index,
        calib_buffer=calib_buffer,
        calib_index=calib_index,
        recording_flag=recording_flag,
        recording_gesture=recording_gesture
    )
    
    # Connect to Myos
    myo_driver.run()  # Now returns after connections complete!
    # Verify both Myos have device names
    if len(myo_driver.myos) >= 2:
        for i, myo in enumerate(myo_driver.myos):
            if myo.device_name is None:
                print(f"✗ Warning: Myo {i} (connection {myo.connection_id}) has no device name!")
            else:
                print(f"✓ Myo {i}: {myo.device_name} (connection {myo.connection_id})")
    print("Data acquisition process ready!")
    
    # Run forever - NOW this works correctly
    while True:
        myo_driver.receive()
        
def get_calibration_buffer_from_shared_mem(calib_buffer, calib_index):
    """Read recorded calibration data"""
    # Return only the filled portion of the buffer
    num_samples = calib_index.value
    if num_samples == 0:
        return None
    return calib_buffer[:num_samples].copy()

def get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION):
    """Read the last N seconds from the streaming buffer"""
    # Calculate how many samples we need
    samples_per_second = 100  # Approximate (depends on actual rate)
    num_samples = int(window_seconds * samples_per_second)
    
    # Get current position
    current_idx = stream_index.value
    
    # Handle wrap-around for circular buffer
    if current_idx < num_samples:
        # Buffer hasn't filled yet
        return stream_buffer[:current_idx].copy()
    else:
        # Calculate wrapped indices
        buffer_size = len(stream_buffer)
        start_idx_wrapped = (current_idx - num_samples) % buffer_size
        current_idx_wrapped = current_idx % buffer_size
        
        # Check if we need to wrap around
        if start_idx_wrapped < current_idx_wrapped:
            # No wrap-around needed
            return stream_buffer[start_idx_wrapped:current_idx_wrapped].copy()
        else:
            # Wrap-around: get two parts and concatenate in correct order
            part1 = stream_buffer[start_idx_wrapped:]  # From start to end of buffer
            part2 = stream_buffer[:current_idx_wrapped]  # From beginning to current
            return np.concatenate([part1, part2])
        
def Calibrate(gesture_name, calib_buffer, calib_index, recording_flag, 
              recording_gesture, classifier):
    """Called from main process when user wants to calibrate"""
    """print(f"Calibration will start in ", end='', flush=True)
    for i in range(CALIBRATION_STARTS, 0, -1):
        print(f"{i}... ", end='', flush=True)
        time.sleep(1)
    print("\n")
    print(f"Recording calibration for '{gesture_name}' - '{CALIBRATION_DURATION} seconds...")
    """ 
    #şimdilik veri toplayacağumuz için burası kapalı.
    # Reset calibration buffer
    calib_index.value = 0
    
    # Set flag in shared memory
    gesture_bytes = gesture_name.encode('utf-8')
    for i, byte in enumerate(gesture_bytes[:50]):  # Max 50 chars
        recording_gesture[i] = byte
    recording_flag.value = 1  # Start recording
    
    # Wait 3 seconds
    print("Recording... ", end='', flush=True)
    for i in range(CALIBRATION_DURATION):
        time.sleep(1)
        print(f"{CALIBRATION_DURATION-i}... ", end='', flush=True)
    print("Done!")
    
    # Stop recording
    recording_flag.value = 0
    time.sleep(0.5)  # Let final batch flush
    # Read recorded data
    recorded_data = get_calibration_buffer_from_shared_mem(calib_buffer, calib_index)
    
    if recorded_data is None or len(recorded_data) == 0:
        print("ERROR: No data was recorded! Check if Myos are connected.")
        return
    
    # Add to classifier
    classifier.add_calibration_sample(gesture_name, recorded_data)
    
    # Save to disk
    import os
    os.makedirs('calibration_data', exist_ok=True)
    timestamp = int(time.time())
    np.save(f'calibration_data/{gesture_name}_{timestamp}.npy', recorded_data)
    
    print(f"Calibration complete! Saved {len(recorded_data)} samples")

def Classify(stream_mem_name, stream_index, is_running_flag, result_queue):
    """Process 2: Runs classification separately"""
    import time
    import numpy as np
    from multiprocessing import shared_memory
    
    # Attach to shared memory
    shm_stream = shared_memory.SharedMemory(name=stream_mem_name)
    stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_stream.buf)
    
    # Load both models
    rest_model = RestDetector(window_size=20, threshold_factor=6.0, min_duration=20, padding=0)
    if not rest_model.load_model('rest_model.pkl'):
        print("ERROR: Could not load rest_model.pkl")
        return
    
    gesture_model = GestureModel(window_size_ms=100, sampling_rate=200)
    if not gesture_model.load_model('gesture_model.pkl'):
        print("ERROR: Could not load gesture_model.pkl")
        return
    
    print("Classification process ready!")
    
    last_position = 0
    # Calculate window sizes
    # RestDetector works on raw data windows (20ms = 4 samples at 200Hz)
    rest_window_samples = 4  # 20ms * 200Hz
    # GestureModel uses 100ms windows (20 samples at 200Hz)
    gesture_window_samples = gesture_model.samples_per_window  # Should be 20
    while True:
        if is_running_flag.value == 0:
            time.sleep(0.01)
            continue
        
        # Wait for new data
        current_position = stream_index.value
        if current_position <= last_position:
            time.sleep(0.001)
            continue
        
        # Check if we have enough data for 20ms window (4 samples)
        if current_position < rest_window_samples:
            continue
        
        # Get last 20ms of data (4 samples)
        end_idx = current_position % STREAM_BUFFER_SIZE
        start_idx = (current_position - rest_window_samples) % STREAM_BUFFER_SIZE

        if start_idx < end_idx:
            window_20ms = stream_buffer[start_idx:end_idx].copy()
        else:
            # Wrap around
            part1 = stream_buffer[start_idx:]
            part2 = stream_buffer[:end_idx]
            window_20ms = np.concatenate([part1, part2])
        
        # Extract features and check if rest
        features_20ms = rest_model.extract_features(window_20ms)
        
        is_rest = rest_model.predict(window_20ms)
        if is_rest:
            last_position = current_position
            continue  # Rest position, skip classification
        
        # Not rest - check if we have enough data for 100ms window (20 samples)
        if current_position < gesture_model.samples_per_window:
            continue
        
        # Get last 100ms of data (20 samples)
        end_idx_100 = current_position % STREAM_BUFFER_SIZE
        start_idx_100 = (current_position - gesture_model.samples_per_window) % STREAM_BUFFER_SIZE
        
        if start_idx_100 < end_idx_100:
            window_100ms = stream_buffer[start_idx_100:end_idx_100].copy()
        else:
            # Wrap around
            part1 = stream_buffer[start_idx_100:]
            part2 = stream_buffer[:end_idx_100]
            window_100ms = np.concatenate([part1, part2])
        
        # Extract features and classify
        features_100ms = gesture_model.extract_features(window_100ms)
        result = gesture_model.classify(features_100ms)
        
        # Send result to main process
        result_queue.put(result)
        print(f"🎯 Detected: {result}")
        
        last_position = current_position

def Train():
    """Train both models - RestModel on ALL participants, GestureModel on segmented data"""
    import glob
    import os
    
    print("=== Training Models ===")
    
    # Train RestModel on ALL participants' rest data (combined)
    print("\n1. Training RestModel on ALL participants' rest data...")
    rest_data = []
    for participant_id in range(1, 7):
        folder = f'calibration_data/p{participant_id}rest'
        if os.path.exists(folder):
            files = glob.glob(f'{folder}/*.npy')
            for file in files:
                data = np.load(file)
                rest_data.append(data)
                print(f"  ✓ Loaded: {os.path.basename(file)} from participant {participant_id}")
    
    if len(rest_data) == 0:
        print("ERROR: No rest data found!")
        return False
    
    print(f"\nTraining on {len(rest_data)} rest samples from all participants...")
    rest_model = RestDetector(window_size=20, threshold_factor=6.0, min_duration=20, padding=0)
    rest_model.fit(rest_data)
    rest_model.save_model('rest_model.pkl')
    print("✓ RestModel saved as rest_model.pkl (for real-time use)")
    
    # Train GestureModel on segmented data from pXnew folders
    print("\n2. Training GestureModel on segmented gesture data...")
    gesture_data = {}
    for participant_id in range(1, 7):
        folder = f'processed_data/p{participant_id}'
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping...")
            continue
        
        files = glob.glob(f'{folder}/*.npy')
        for file in files:
            basename = os.path.basename(file)
            gesture_name = basename.split('_')[0]
            
            if gesture_name not in gesture_data:
                gesture_data[gesture_name] = []
            
            gesture_data[gesture_name].append(np.load(file))
    
    
    print(f"Found {len(gesture_data)} gesture types:")
    for gesture_name, samples in gesture_data.items():
        print(f"  - {gesture_name}: {len(samples)} samples")
    
    gesture_model = GestureModel(window_size_ms=100, sampling_rate=200)
    gesture_model.train(gesture_data)
    gesture_model.save_model('gesture_model.pkl')
    print("✓ GestureModel saved as gesture_model.pkl")
    
    print("\n=== Training Complete! ===")
    print("Models ready for real-time classification")
    return True

def Command(stream_buffer, stream_index, calib_buffer, calib_index, 
           recording_flag, recording_gesture, classifier, is_running_flag): 
    value = input("Enter your command majesty: ")
    match value:
        case "tr":  #train
            print("now will run train function")
            Train()
        case "cb":  #calibrate
            print("now will run calibrate function")
            gesture_name = input("Which gesture would you like to calibrate? ")
            Calibrate(gesture_name, calib_buffer, calib_index, recording_flag, 
                     recording_gesture, classifier)
        case "startcf":
            print("starting classification")
            is_running_flag.value = 1
        case "stopcf":
            print("stopping classification")
            is_running_flag.value = 0
        case _:
            print("Invalid command! Use: tr, cf, cb, or live")

if __name__ == "__main__":
    print("=== Gesture Recognition System ===")
    print("Initializing...")
    
    # Create shared memory buffers
    shm_stream = shared_memory.SharedMemory(create=True, size=STREAM_BUFFER_SIZE*34*4)
    stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_stream.buf)
    stream_buffer.fill(0)  # Initialize to zero
    
    shm_calib = shared_memory.SharedMemory(create=True, size=CALIBRATION_BUFFER_SIZE*34*4)
    calib_buffer = np.ndarray((CALIBRATION_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_calib.buf)
    calib_buffer.fill(0)
    
    # Create shared indices and flags
    stream_index = mp.Value('i', 0)
    calib_index = mp.Value('i', 0)
    recording_flag = mp.Value('i', 0)
    recording_gesture = mp.Array('c', 50)
    is_running_flag = mp.Value('i', 0)

    # Start data acquisition process
    data_process = Process(
        target=data_acquisition_process,
        args=(shm_stream.name, shm_calib.name, stream_index, calib_index, 
              recording_flag, recording_gesture)
    )
    data_process.daemon = True  # Dies when main process dies
    data_process.start()
    time.sleep(2)  # wait for dataprocess to start
    # Give it time to connect
    print("Connecting to Myo armbands...")
    time.sleep(5)

    if not data_process.is_alive():
        print("ERROR: Data acquisition process failed to start!")
        print("Check if Myo dongle is connected and MyoConnect is closed")
    
    # Give it time to connect
    print("Connecting to Myo armbands...")
    time.sleep(5)
    # Start classification process
    classify_process = Process(
        target=Classify,
        args=(shm_stream.name, stream_index, is_running_flag)  # No result_queue
    )
    classify_process.daemon = True
    classify_process.start()
    
    print("\nSystem ready! Available commands:")
    print("  tr = train models")
    print("  cb = calibrate gesture")
    print("  startcf = start classification")
    print("  stopcf = stop classification")
    print()
    # Command loop
    try:
        while True:
            Command(stream_buffer, stream_index, calib_buffer, calib_index,
                   recording_flag, recording_gesture, is_running_flag)
    except KeyboardInterrupt:
        print("\nShutting down...")
        is_running_flag.value = 0
        data_process.terminate()
        data_process.join()
        shm_stream.close()
        shm_stream.unlink()
        shm_calib.close()
        shm_calib.unlink()
        print("Goodbye!")
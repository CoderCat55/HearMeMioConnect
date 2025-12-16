"""handles multiprocessing and user commands"""
import multiprocessing as mp
from multiprocessing import Process, shared_memory, freeze_support
import numpy as np
from model import GestureClassifier
import time

# Constants
STREAM_BUFFER_SIZE = 1000  # ~5 seconds at 200Hz
CALIBRATION_BUFFER_SIZE = 600  # 3 seconds at 200Hz

CALIBRATION_DURATION = 3  # seconds
CLASSIFICATION_DURATION = 3  # same!

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
    # The Myo armband's EMG data rate is 200Hz. This is a critical value.
    samples_per_second = 100
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

def Classify(stream_buffer, stream_index, classifier):
    """Called from main process when user wants to classify"""
    time.sleep(CLASSIFICATION_DURATION)
    # Read current data from shared memory (last CLASSIFICATION_DURATION second)
    current_data = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION)
    
    if current_data is None or len(current_data) < 10:
        print("ERROR: Not enough data to classify!")
        return None
    
    # Extract features
    features = GestureClassifier.extract_features(current_data)
    
    # Classify
    result = classifier.classify(features)
    print(f"Predicted gesture: {result}")
    return result

def LiveClassify(stream_buffer, stream_index, classifier):
    """Continuously classify gestures in real-time for a fixed number of iterations."""
    if not classifier.model:
        print("ERROR: Model not loaded. Please train a model using 'tr' or ensure model files are present.")
        return

    print("\n>>> Starting Standard LIVE classification (20 iterations) <<<")
    last_gesture = "---"
    for i in range(1, 21):
        # Get recent data
        current_data = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION)
    
        if current_data is None or len(current_data) < 10:
            print(f"\rIteration {i}/20: Not enough data...", end="", flush=True)
            time.sleep(0.5)
            continue

        # Extract features and classify
        features = GestureClassifier.extract_features(current_data)
        prediction = classifier.classify(features)

        # Only update if the prediction is confident
        if prediction not in ["Uncertain", "Not enough data"]:
            last_gesture = prediction
        
        print(f"\rIteration {i}/20: {last_gesture.ljust(20)}", end="", flush=True)
        time.sleep(0.5) # Pause between iterations
    print("\n>>> Live classification finished. <<<")

def ContinuousLiveClassify(stream_buffer, stream_index, classifier):
    """Continuously classify gestures in real-time with stateful smoothing until interrupted."""
    if not classifier.model:
        print("ERROR: Model not loaded. Please use 'tr' or 'load' command.")
        return

    print("\n>>> Starting CONTINUOUS LIVE classification... Press Ctrl+C to stop. <<<")
    last_stable_gesture = "---"
    try:
        while True:
            current_data = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION)
            if current_data is None or len(current_data) < 10:
                time.sleep(0.5)
                continue
            features = GestureClassifier.extract_features(current_data)
            prediction = classifier.classify(features)
            if prediction not in ["Uncertain", "Not enough data", "---"] and prediction != last_stable_gesture:
                last_stable_gesture = prediction
            
            print(f"\rPrediction: {last_stable_gesture.ljust(20)}", end="", flush=True)
            time.sleep(0.2) # Shorter sleep for more responsive feel
    except KeyboardInterrupt:
        print("\n>>> Live classification stopped. <<<")

def LiveValidate(stream_buffer, stream_index, classifier):
    """Runs a structured validation test to measure real-time accuracy."""
    if not classifier.model or not classifier.gesture_map:
        print("ERROR: Model not loaded. Please use 'tr' or 'load' command.")
        return

    gestures_to_test = list(classifier.gesture_map.keys())
    if not gestures_to_test:
        print("No gestures found in the model map.")
        return

    print("\n--- Starting Live Validation Mode ---")
    print("For each gesture, you will be asked to perform it for 5 seconds.")
    
    overall_results = {}

    for gesture in gestures_to_test:
        input(f"\nPress Enter when you are ready to perform the '{gesture}' gesture...")
        print(f"Performing '{gesture}' for 5 seconds... ", end="", flush=True)
        
        predictions = []
        start_time = time.time()
        while time.time() - start_time < 5:
            current_data = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION)
            if current_data is None or len(current_data) < 200: # Need at least 1s of data
                time.sleep(0.1)
                continue
            
            features = GestureClassifier.extract_features(current_data)
            prediction = classifier.classify(features)
            predictions.append(prediction)
            time.sleep(0.2)
        
        # Calculate and display results for this gesture
        correct_count = predictions.count(gesture)
        accuracy = (correct_count / len(predictions)) * 100 if predictions else 0
        print(f"Done! Real-time accuracy for '{gesture}': {accuracy:.2f}%")
        overall_results[gesture] = accuracy

    print("\n--- Live Validation Summary ---")
    for gesture, acc in overall_results.items():
        print(f"- {gesture.ljust(15)}: {acc:.2f}%")
    print("---------------------------------")

def Train(classifier):
    """Called from main process to train a new model using the internal method."""
    print("--- Initializing model training ---")
    success = classifier.train()
    if success:
        print("--- Model successfully trained and loaded! ---")
    else:
        print("--- Model training failed. Please check logs for errors. ---")

def LoadModel(classifier):
    """Presents a menu to the user to load a specific trained model."""
    available_models = {
        "1": "decision_tree",
        "2": "logistic_regression",
        "3": "k_nearest_neighbors",
        "4": "support_vector_machine",
        "5": "random_forest"
    } # Removed Gradient Boosting

    print("\n--- Available Models ---")
    for key, name in available_models.items():
        # Format the name for display, e.g., "decision_tree" -> "Decision Tree"
        display_name = name.replace('_', ' ').title()
        print(f"  {key}: {display_name}")

    choice = input("Select a model to load (or press Enter to cancel): ")

    if choice in available_models:
        model_name = available_models[choice]
        model_path = f'model_{model_name}.pkl'
        print(f"\nAttempting to load '{model_path}'...")
        # The scaler and map are common, so we only need to specify the new model path.
        classifier.load_model(model_path=model_path)
    elif choice == "":
        print("Model loading cancelled.")
    else:
        print("Invalid selection.")

def Command(stream_buffer, stream_index, calib_buffer, calib_index, 
           recording_flag, recording_gesture, classifier):
    prompt = f"[{classifier.loaded_model_name}] Enter your command: "
    value = input(prompt) 
    match value:
        case "tr":  #train
            print("now will run train function")
            Train(classifier)
        case "cf": # classify <3
            print("now will run classify function")
            print(f"Classify will start in ", end='', flush=True)
            for i in range(CLASSIFICATION_STARTS, 0, -1):
                print(f"{i}... ", end='', flush=True)
                time.sleep(1)
            print("\n")
            print("Classifying gesture...")
            Classify(stream_buffer, stream_index, classifier)
        case "cb":  #calibrate
            print("now will run calibrate function")
            gesture_name = input("Which gesture would you like to calibrate? ")
            Calibrate(gesture_name, calib_buffer, calib_index, recording_flag, 
                     recording_gesture, classifier)
        case "live": # live classification
            print("Starting standard live classification (20 iterations)...")
            print(f"Waiting for buffer to fill... ", end='', flush=True)
            for i in range(CLASSIFICATION_STARTS, 0, -1):
                print(f"{i}... ", end='', flush=True)
                time.sleep(1)
            print("\n")
            LiveClassify(stream_buffer, stream_index, classifier)
        case "clive": # continuous live classification
            print("Starting continuous live classification...")
            ContinuousLiveClassify(stream_buffer, stream_index, classifier)
        case "validate": # Run the live validation test
            LiveValidate(stream_buffer, stream_index, classifier)
        case "load": # load a specific model
            LoadModel(classifier)
        case _:
            print("Invalid command! Use: tr, cf, cb, live, clive, validate, load")

if __name__ == "__main__":
    # Required for multiprocessing on Windows when building an executable
    freeze_support() 

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
    
    # Initialize classifier in main process
    classifier = GestureClassifier()
    classifier.load_model() # Try to load existing model on startup
    
    print("\nSystem ready! Available commands: tr, cf, cb, live, clive, validate, load")
    
    # Command loop
    try:
        while True:
            Command(stream_buffer, stream_index, calib_buffer, calib_index,
                   recording_flag, recording_gesture, classifier)
    except KeyboardInterrupt:
        print("\nShutting down...")
        data_process.terminate()
        data_process.join()
        shm_stream.close()
        shm_stream.unlink()
        shm_calib.close()
        shm_calib.unlink()
        print("Goodbye!")

"""handles multiprocessing and user commands"""
import multiprocessing as mp
from multiprocessing import Process, shared_memory
import numpy as np
import time
from rest_model import RestDetector
from gesture_model import GestureModel
import os
import threading


# Constants
SAMPLINGHZ= 50
# Calculate window sizes
rest_window_samples = 20  #note if you are gonna change this change it also in classify function
gesture_window_samples = 200

STREAM_BUFFER_SIZE = 1000  # ~5 seconds at SAMPLINGHZ
CALIBRATION_BUFFER_SIZE = 250  # ~3 seconds at SAMPLINGHZ

CALIBRATION_DURATION = 3  
CLASSIFICATION_DURATION = 3 

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

def get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION,sampling_rate=SAMPLINGHZ):
    """Read the last N seconds from the streaming buffer"""
    # Calculate how many samples we need
    num_samples = int(window_seconds * sampling_rate)
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
        
def Calibrate(gesture_name, calib_buffer, calib_index, recording_flag, recording_gesture):
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
    #classifier.add_calibration_sample(gesture_name, recorded_data)
    
    # Save to disk
    import os
    os.makedirs('calibration_data', exist_ok=True)
    timestamp = int(time.time())
    np.save(f'calibration_data/{gesture_name}_{timestamp}.npy', recorded_data)
    
    print(f"Calibration complete! Saved {len(recorded_data)} samples")
"""Calibrate funcitoan will be dealt with later"""

def Classify(stream_mem_name, stream_index, is_running_flag, result_queue, STREAM_BUFFER_SIZE, samplingrate):
    """Process 2: Runs classification using Rest-to-Rest strategy"""
    import time
    import numpy as np
    from multiprocessing import shared_memory
    import requests
    
    # Calculate window sizes
    REST_WINDOW_SIZE = 20  
    
    # Attach to shared memory
    shm_stream = shared_memory.SharedMemory(name=stream_mem_name)
    stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_stream.buf)
    result_queue.put("CLASSIFY: Shared memory attached")
    
    # Load both models
    rest_model = RestDetector(window_size=REST_WINDOW_SIZE)
    if not rest_model.load_model('rest_model.pkl'):
        result_queue.put("ERROR: Could not load rest_model.pkl")
        return
    
    # Note: We still load GestureModel, but we will feed it variable length data now
    gesture_model = GestureModel(window_size_samples=200, sampling_rate=samplingrate)
    if not gesture_model.load_model('gesture_model.pkl'):
        result_queue.put("ERROR: Could not load gesture_model.pkl")
        return
    
    result_queue.put("✓ Classification process ready! (Waiting for Non-Rest to Rest sequence)")
    
    last_processed_idx = 0
    gesture_start_idx = None  # State variable: None = Waiting, Int = Recording
    
    while True:
        if is_running_flag.value == 0:
            time.sleep(0.01)
            continue
        
        # 1. Get current absolute position
        current_idx = stream_index.value
        
        # 2. Check if we have enough new data for a rest check
        if current_idx - last_processed_idx < 1: # check often
             time.sleep(0.002)
             continue

        # Ensure we have at least one window of data to check rest
        if current_idx < REST_WINDOW_SIZE:
            continue

        # 3. Extract the latest small window to check if hand is resting
        # Handle wrap around for the Rest Window
        r_end = current_idx % STREAM_BUFFER_SIZE
        r_start = (current_idx - REST_WINDOW_SIZE) % STREAM_BUFFER_SIZE
        
        if r_start < r_end:
            rest_check_data = stream_buffer[r_start:r_end]
        else:
            rest_check_data = np.concatenate([stream_buffer[r_start:], stream_buffer[:r_end]])

        # 4. Predict State
        is_rest = rest_model.predict(rest_check_data)
        
        # --- STATE MACHINE LOGIC ---
        
        # State A: We are currently WAITING for a gesture to start
        if gesture_start_idx is None:
            if not is_rest:
                # TRANSITION: Rest -> Active
                # We assume the gesture started a bit before the detector triggered, 
                # so we set start index back by the window size.
                gesture_start_idx = current_idx - REST_WINDOW_SIZE
                result_queue.put(f"⚡ Movement started at idx {gesture_start_idx}...")
        
        # State B: We are currently RECORDING a gesture
        else:
            if is_rest:
                # TRANSITION: Active -> Rest (Gesture Finished)
                gesture_end_idx = current_idx
                duration_samples = gesture_end_idx - gesture_start_idx
                
                # Filter: Ignore very short blips (e.g., < 10 samples)
                if duration_samples > 10:
                    result_queue.put(f"🛑 Movement ended. Capturing {duration_samples} samples...")
                    
                    # Extract the FULL gesture from shared memory
                    # We must handle wrap around for the potentially large chunk
                    g_start_wrapped = gesture_start_idx % STREAM_BUFFER_SIZE
                    g_end_wrapped = gesture_end_idx % STREAM_BUFFER_SIZE
                    
                    if g_start_wrapped < g_end_wrapped:
                        full_gesture_data = stream_buffer[g_start_wrapped:g_end_wrapped].copy()
                    else:
                        full_gesture_data = np.concatenate([
                            stream_buffer[g_start_wrapped:], 
                            stream_buffer[:g_end_wrapped]
                        ])
                    
                    # CLASSIFY
                    # Note: full_gesture_data size is variable now. 
                    # extract_features handles this, but model accuracy depends on training data.
                    features = gesture_model.extract_features(full_gesture_data)
                    result = gesture_model.classify(features)
                    
                    result_queue.put(f"🎯 RESULT: {result}")
                    # Send result to webserver
                    try:
                        requests.get(
                            f'http://localhost:5002/result?value={result}', 
                            timeout=0.5
                        )
                    except Exception as e:
                        pass  # Don't crash if webserver is unreachable
                else:
                    result_queue.put("⚠️ Ignored short blip")

                # Reset state to waiting
                gesture_start_idx = None
            
            # If still not rest, we just loop and wait for more data to accumulate.
            # Safety check: If gesture gets too long (buffer overflow risk), force stop
            elif (current_idx - gesture_start_idx) >= STREAM_BUFFER_SIZE:
                result_queue.put("⚠️ Buffer Overflow: Gesture too long, resetting.")
                gesture_start_idx = None

        last_processed_idx = current_idx

def Train():
    #Train both models - RestModel on ALL participants, GestureModel on segmented data"""
    import glob
    import os
    
    print("=== Training Models ===")
    rest_model = RestDetector(window_size=rest_window_samples)
    rest_model.train() #burada datalar rest_model.py tarafından çağrıldığı için gesture_model kadar kod yok.
    rest_model.save_model('rest_model.pkl')
    print("✓ RestModel saved as rest_model.pkl (for real-time use)")
    
    # Train GestureModel on segmented data from participant folders
    print("\n2. Training GestureModel on segmented gesture data...")
    gesture_data = {}
    for participant_id in range(1, 7):
        folder = f'rows_deleted/p{participant_id}'
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping...")
            continue
        files = glob.glob(f'{folder}/*.npy')
        # FILTER: only files NOT starting with "rest"
        files = [f for f in files if not os.path.basename(f).startswith('rest')]
        for file in files:
            basename = os.path.basename(file)
            gesture_name = basename.split('_')[0]
            
            if gesture_name not in gesture_data:
                gesture_data[gesture_name] = []
            
            gesture_data[gesture_name].append(np.load(file))
    
    
    print(f"Found {len(gesture_data)} gesture types:")
    for gesture_name, samples in gesture_data.items():
        print(f"  - {gesture_name}: {len(samples)} samples")
    
    gesture_model = GestureModel(window_size_samples=gesture_window_samples, sampling_rate=SAMPLINGHZ)
    gesture_model.train(gesture_data)
    gesture_model.save_model('gesture_model.pkl')
    print("✓ GestureModel saved as gesture_model.pkl")
    
    print("\n=== Training Complete! ===")
    print("Models ready for real-time classification")
    return True
def Command(stream_buffer, stream_index, calib_buffer, calib_index, 
           recording_flag, recording_gesture, is_running_flag, system): 
    value = input("Enter your command majesty: ")
    match value:
        case "connect":
            print("Starting data acquisition...")
            system.start_data_acquisition()
        case "disconnect":
            print("Stopping data acquisition...")
            system.stop_data_acquisition()
        case "tr":  #train
            print("now will run train function")
            Train()
        case "cb":  #calibrate
            if not system.is_data_acquisition_running():
                print("ERROR: Data acquisition not running. Use 'connect' first.")
            else:
                print("now will run calibrate function")
                gesture_name = input("Which gesture would you like to calibrate? ")
                Calibrate(gesture_name, calib_buffer, calib_index, recording_flag, 
                         recording_gesture)
        case "startcf":
            if not system.is_data_acquisition_running():
                print("ERROR: Data acquisition not running. Use 'connect' first.")
            else:
                print("starting classification")
                is_running_flag.value = 1
        case "stopcf":
            print("stopping classification")
            is_running_flag.value = 0
        case "debug":
            print(f"Data acquisition running: {system.is_data_acquisition_running()}")
            print(f"Stream index: {stream_index.value}")
            print(f"Classification running: {is_running_flag.value}")
            if stream_index.value > 0:
                print(f"Recent data sample: {stream_buffer[stream_index.value % STREAM_BUFFER_SIZE][:8]}")
        case _:
            print("Invalid command! Use: connect, disconnect, tr, cb, startcf, stopcf, debug")

def monitor_classification_results(result_queue):
    """Read from queue and print to terminal"""
    while True:
        try:
            message = result_queue.get(timeout=0.1)
            print(f"\n{message}")  # Print with newline so it doesn't mess up input prompt
        except:
            continue  # Queue empty, keep checking

class GestureSystem:
    """Centralized system controller for webserver integration"""
    
    def __init__(self):
        print("Initializing GestureSystem...")
        
        # Shared memory buffers
        self.shm_stream = None
        self.shm_calib = None
        self.stream_buffer = None
        self.calib_buffer = None
        
        # Shared control variables
        self.stream_index = None
        self.calib_index = None
        self.recording_flag = None
        self.recording_gesture = None
        self.is_running_flag = None
        self.result_queue = None
        
        # Process/thread handles
        self.data_process = None
        self.classify_process = None
        self.monitor_thread = None
        
        # Models (for status checking)
        self.rest_model = None
        self.gesture_model = None
        
        self._initialize_shared_memory()
        self._load_models()
    
    def _initialize_shared_memory(self):
        """Create all shared memory structures"""
        # Stream buffer
        self.shm_stream = shared_memory.SharedMemory(
            create=True, 
            size=STREAM_BUFFER_SIZE * 34 * 4
        )
        self.stream_buffer = np.ndarray(
            (STREAM_BUFFER_SIZE, 34), 
            dtype=np.float32, 
            buffer=self.shm_stream.buf
        )
        self.stream_buffer.fill(0)
        
        # Calibration buffer
        self.shm_calib = shared_memory.SharedMemory(
            create=True, 
            size=CALIBRATION_BUFFER_SIZE * 34 * 4
        )
        self.calib_buffer = np.ndarray(
            (CALIBRATION_BUFFER_SIZE, 34), 
            dtype=np.float32, 
            buffer=self.shm_calib.buf
        )
        self.calib_buffer.fill(0)
        
        # Shared indices and flags
        self.stream_index = mp.Value('i', 0)
        self.calib_index = mp.Value('i', 0)
        self.recording_flag = mp.Value('i', 0)
        self.recording_gesture = mp.Array('c', 50)
        self.is_running_flag = mp.Value('i', 0)
        self.result_queue = mp.Queue()
        
        print("✓ Shared memory initialized")
    
    def _load_models(self):
        """Load trained models for status checking"""
        try:
            self.rest_model = RestDetector(window_size=rest_window_samples)
            if os.path.exists('rest_model.pkl'):
                self.rest_model.load_model('rest_model.pkl')
            
            self.gesture_model = GestureModel(
                window_size_samples=gesture_window_samples, 
                sampling_rate=SAMPLINGHZ
            )
            if os.path.exists('gesture_model.pkl'):
                self.gesture_model.load_model('gesture_model.pkl')
        except Exception as e:
            print(f"Note: Models not loaded yet ({e})")
    
    def start_data_acquisition(self):
        """Start the data acquisition process"""
        if self.data_process is not None and self.data_process.is_alive():
            print("Data acquisition already running")
            return True
        
        self.data_process = Process(
            target=data_acquisition_process,
            args=(
                self.shm_stream.name, 
                self.shm_calib.name, 
                self.stream_index, 
                self.calib_index,
                self.recording_flag, 
                self.recording_gesture
            )
        )
        self.data_process.daemon = True
        self.data_process.start()
        
        # Wait for initialization
        time.sleep(5)
        
        if self.data_process.is_alive():
            print("✓ Data acquisition started")
            return True
        else:
            print("✗ Data acquisition failed to start")
            return False
    
    def stop_data_acquisition(self):
        """Stop the data acquisition process"""
        if self.data_process is not None and self.data_process.is_alive():
            self.data_process.terminate()
            self.data_process.join(timeout=2)
            print("✓ Data acquisition stopped")
    
    def is_data_acquisition_running(self):
        """Check if data acquisition is running"""
        return self.data_process is not None and self.data_process.is_alive()
    
    def start_classification_process(self):
        """Start the classification process (called once at startup)"""
        if self.classify_process is not None and self.classify_process.is_alive():
            print("Classification process already exists")
            return True
        
        self.classify_process = Process(
            target=Classify,
            args=(
                self.shm_stream.name, 
                self.stream_index, 
                self.is_running_flag,
                self.result_queue,
                STREAM_BUFFER_SIZE,
                SAMPLINGHZ
            )
        )
        self.classify_process.daemon = True
        self.classify_process.start()
        time.sleep(1)
        
        print("✓ Classification process started (paused)")
        return True
    
    def start_classification(self):
        """Enable classification (set flag to 1)"""
        self.is_running_flag.value = 1
        print("✓ Classification enabled")
    
    def stop_classification(self):
        """Disable classification (set flag to 0)"""
        self.is_running_flag.value = 0
        print("✓ Classification disabled")
    
    def is_classification_running(self):
        """Check if classification is active"""
        return bool(self.is_running_flag.value)
    
    def start_monitor_thread(self):
        """Start thread to monitor classification results"""
        self.monitor_thread = threading.Thread(
            target=monitor_classification_results,
            args=(self.result_queue,),
            daemon=True
        )
        self.monitor_thread.start()
        print("✓ Monitor thread started")
    
    def calibrate(self, gesture_name):
        """Calibrate a gesture"""
        if not self.is_data_acquisition_running():
            print("ERROR: Data acquisition not running")
            return False
        
        try:
            Calibrate(
                gesture_name, 
                self.calib_buffer, 
                self.calib_index,
                self.recording_flag, 
                self.recording_gesture
            )
            return True
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False
    
    def train_models(self):
        """Train both models"""
        try:
            success = Train()
            if success:
                self._load_models()  # Reload models after training
            return success
        except Exception as e:
            print(f"Training failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up...")
        
        # Stop classification
        self.is_running_flag.value = 0
        
        # Terminate processes
        if self.data_process is not None:
            self.data_process.terminate()
            self.data_process.join(timeout=2)
        
        if self.classify_process is not None:
            self.classify_process.terminate()
            self.classify_process.join(timeout=2)
        
        # Clean up shared memory
        try:
            self.shm_stream.close()
            self.shm_stream.unlink()
        except:
            pass
        
        try:
            self.shm_calib.close()
            self.shm_calib.unlink()
        except:
            pass
        
        print("✓ Cleanup complete")

if __name__ == "__main__":
    print("=" * 50)
    print("=== Gesture Recognition System ===")
    print("=" * 50)
    
    # Create system instance
    system = GestureSystem()
    # Start classification process (it will wait for start command)
    print("Starting classification process...")
    system.start_classification_process()
    
    # Start monitor thread for terminal output
    system.start_monitor_thread()
    
    # ====== Start Webserver ======
    print("\nStarting webserver...")
    from webserver import app, inject_system
    inject_system(system)
    
    webserver_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False),
        daemon=True
    )
    webserver_thread.start()
    print(f"✓ Webserver running on http://0.0.0.0:5002")
    print("=" * 50)
    
    print("\nSystem ready! Available commands:")
    print("  tr       = train models")
    print("  cb       = calibrate gesture")
    print("  startcf  = start classification")
    print("  stopcf   = stop classification")
    print("  debug    = show debug info")
    print()
    
    # Command loop
    try:
        while True:
            Command(
                system.stream_buffer, 
                system.stream_index, 
                system.calib_buffer, 
                system.calib_index,
                system.recording_flag, 
                system.recording_gesture,
                system.is_running_flag,
                system
            )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        system.cleanup()
        print("Goodbye!")
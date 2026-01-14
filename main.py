"""handles multiprocessing and user commands"""
import multiprocessing as mp
from multiprocessing import Process, shared_memory
import numpy as np
import time
from rest_model import RestDetector
from gesture_model import GestureModel
from personal_model import PersonalModel
import os
import threading


# Constants
SAMPLINGHZ= 50
# Calculate window sizes
rest_window_samples = 20  #note if you are gonna change this change it also in classify function
gesture_window_samples = 75

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
        
def Calibrate(gesture_name, stream_buffer, stream_index, calib_buffer, calib_index, 
              recording_flag, recording_gesture):
    """
    Calibrate using rest-to-rest detection (like classification)
    Waits for: Rest -> Movement -> Rest, then saves the gesture
    """
    # Load rest model
    rest_model = RestDetector(window_size=rest_window_samples)
    if not rest_model.load_model('rest_model.pkl'):
        print("ERROR: rest_model.pkl not found! Please train models first (command: tr)")
        return False
    
    REST_WINDOW_SIZE = rest_window_samples
    MIN_GESTURE_SAMPLES = 10  # Same as classification
    
    print(f"\n{'='*50}")
    print(f"CALIBRATION MODE: {gesture_name}")
    print(f"{'='*50}")
    print("Instructions:")
    print("  1. Keep your hand at REST position")
    print("  2. When ready, perform the gesture")
    print("  3. Return to REST position")
    print("  4. System will auto-detect and save")
    print(f"{'='*50}\n")
    
    print("⏳ Waiting for REST state...")
    
    last_processed_idx = stream_index.value
    gesture_start_idx = None
    state = "WAITING_FOR_REST"  # States: WAITING_FOR_REST, READY, RECORDING
    
    start_time = time.time()
    timeout = 30  # 30 second timeout
    
    while True:
        # Timeout check
        if time.time() - start_time > timeout:
            print("\n❌ TIMEOUT: Calibration cancelled (30s limit)")
            return False
        
        # Get current position
        current_idx = stream_index.value
        
        # Check if we have enough new data
        if current_idx - last_processed_idx < 1:
            time.sleep(0.002)
            continue
        
        # Need minimum data to check rest
        if current_idx < REST_WINDOW_SIZE:
            time.sleep(0.01)
            continue
        
        # Extract latest window for rest check
        r_end = current_idx % STREAM_BUFFER_SIZE
        r_start = (current_idx - REST_WINDOW_SIZE) % STREAM_BUFFER_SIZE
        
        if r_start < r_end:
            rest_check_data = stream_buffer[r_start:r_end]
        else:
            rest_check_data = np.concatenate([stream_buffer[r_start:], stream_buffer[:r_end]])
        
        # Predict state
        is_rest = rest_model.predict(rest_check_data)
        
        # STATE MACHINE
        if state == "WAITING_FOR_REST":
            if is_rest:
                state = "READY"
                print("✓ REST detected. You may now perform the gesture...")
        
        elif state == "READY":
            if not is_rest:
                # Gesture started!
                gesture_start_idx = current_idx - REST_WINDOW_SIZE
                state = "RECORDING"
                print(f"⚡ Movement detected! Recording started at index {gesture_start_idx}...")
        
        elif state == "RECORDING":
            if is_rest:
                # Gesture ended!
                gesture_end_idx = current_idx
                duration_samples = gesture_end_idx - gesture_start_idx
                
                # Check minimum duration
                if duration_samples < MIN_GESTURE_SAMPLES:
                    print(f"⚠️  Gesture too short ({duration_samples} samples), ignoring. Try again...")
                    state = "READY"
                    gesture_start_idx = None
                    continue
                
                print(f"🛑 Movement ended! Captured {duration_samples} samples")
                
                # Extract the full gesture from stream buffer
                g_start_wrapped = gesture_start_idx % STREAM_BUFFER_SIZE
                g_end_wrapped = gesture_end_idx % STREAM_BUFFER_SIZE
                
                if g_start_wrapped < g_end_wrapped:
                    recorded_data = stream_buffer[g_start_wrapped:g_end_wrapped].copy()
                else:
                    recorded_data = np.concatenate([
                        stream_buffer[g_start_wrapped:], 
                        stream_buffer[:g_end_wrapped]
                    ])
                
                # Save to disk
                os.makedirs('user', exist_ok=True)
                timestamp = int(time.time())
                filepath = f'user/{gesture_name}_{timestamp}.npy'
                np.save(filepath, recorded_data)
                
                print(f"✅ CALIBRATION COMPLETE!")
                print(f"   Saved: {filepath}")
                print(f"   Samples: {len(recorded_data)}")
                print(f"   Duration: {len(recorded_data)/SAMPLINGHZ:.2f}s")
                
                return True
            
            # Safety: Check for buffer overflow
            elif (current_idx - gesture_start_idx) >= STREAM_BUFFER_SIZE:
                print("⚠️  Buffer overflow: Gesture too long, resetting. Try again...")
                state = "READY"
                gesture_start_idx = None
        
        last_processed_idx = current_idx

def Classify(stream_mem_name, stream_index, is_running_flag,Pis_running_flag, result_queue, STREAM_BUFFER_SIZE, samplingrate):
    """Process 2: Runs classification using Rest-to-Rest strategy"""
    """Addition another gesture model that only be trained on data inside 'user' folder """
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
    
    gesture_model = GestureModel(window_size_samples=75, sampling_rate=samplingrate)
    if not gesture_model.load_model('gesture_model.pkl'):
        result_queue.put("ERROR: Could not load gesture_model.pkl it might benot trained yet")
        return
    
    #personal gesture model that will be trained on data in 'user folder'
    Pgesture_model = PersonalModel(window_size_samples=75, sampling_rate=samplingrate)
    if not Pgesture_model.load_model('Pgesture_model.pkl'):
        result_queue.put("ERROR: Could not load Pgesture_model.pkl (personal model not trained yet)")
        Pgesture_model = None  # Mark as unavailable
        return
    
    result_queue.put("✓ Classification process ready! (Waiting for Non-Rest to Rest sequence)")
    
    last_processed_idx = 0
    gesture_start_idx = None  # State variable: None = Waiting, Int = Recording
    
    while True:
        #is_running_flag would control general model that is trained on p1-p6
        #Pis_running_flag would control the personal model that is only trained on data inside 'user' folder

        # Check which model should be active
        using_general_model = (is_running_flag.value == 1)
        using_personal_model = (Pis_running_flag.value == 1)
        
        # If neither is active, sleep and continue
        if not using_general_model and not using_personal_model:
            time.sleep(0.01)
            continue
        
        # Select the active model
        if using_personal_model:
            if Pgesture_model is None:
                result_queue.put("ERROR: Personal model not trained! Please train first.")
                Pis_running_flag.value = 0  # Auto-stop
                continue
            active_model = Pgesture_model
            model_type = "Personal"
        else:
            active_model = gesture_model
            model_type = "General"

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
                    #
                    features = active_model.extract_features(full_gesture_data)
                    result_queue.put(f"DEBUG: features shape = {features.shape}")
                    result_queue.put(f"DEBUG: features sample = {features[:5]}")

                    result = active_model.classify(features)
                    result_queue.put(f"DEBUG: Raw result = '{result}'")
                    result_queue.put(f"DEBUG: Result type = {type(result)}")
                    result_queue.put(f"DEBUG: Result repr = {repr(result)}")

                    # Add model type prefix to differentiate results
                    if model_type == "Personal":
                        result_with_prefix = f"P_{result}"
                    else:
                        result_with_prefix = result

                    result_queue.put(f"🎯 RESULT ({model_type} Model): {result_with_prefix}")

                    # Send result to webserver
                    try:
                        url = f'http://localhost:5002/result?value={result}'
                        result_queue.put(f"DEBUG: Sending URL = {url}")
                        response = requests.get(url, timeout=0.5)
                        result_queue.put(f"DEBUG: Response = {response.json()}")
                    except Exception as e:
                        result_queue.put(f"DEBUG: Request failed: {e}")
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

def TrainPersonal():
    """Sadece 'user' klasöründeki verilerle PersonalModel'i eğitir"""
    import glob
    import os
    
    print("\n=== Training Personal Model (ptr) ===")
    user_folder = 'user'
    
    if not os.path.exists(user_folder):
        print(f"❌ Error: '{user_folder}' folder not found. Please calibrate first (cb).")
        return False
        
    pgesture_data = {}
    files = glob.glob(f'{user_folder}/*.npy')
    # 'rest' ile başlamayan dosyaları filtrele
    files = [f for f in files if not os.path.basename(f).startswith('rest')]
    
    if not files:
        print("❌ Error: No gesture data found in 'user' folder.")
        return False

    for file in files:
        basename = os.path.basename(file)
        gesture_name = basename.split('_')[0]
        if gesture_name not in pgesture_data:
            pgesture_data[gesture_name] = []
        pgesture_data[gesture_name].append(np.load(file))

    print(f"Found {len(pgesture_data)} personal gestures:")
    for name, samples in pgesture_data.items():
        print(f"  - {name}: {len(samples)} samples")

    # Modeli oluştur ve eğit
    p_model = PersonalModel(window_size_samples=gesture_window_samples, sampling_rate=SAMPLINGHZ)
    p_model.train(pgesture_data)
    p_model.save_model('Pgesture_model.pkl')
    print("✓ PersonalModel saved as Pgesture_model.pkl")
    return True

def Train():
    """Sadece genel modelleri (Rest ve Genel Gesture) eğitir"""
    import glob
    import os
    
    print("\n=== Training General Models (tr) ===")
    # 1. Rest Model Eğitimi
    rest_model = RestDetector(window_size=rest_window_samples)
    rest_model.train()
    rest_model.save_model('rest_model.pkl')
    
    # 2. Genel GestureModel Eğitimi (p1-p6 arası veriler)
    gesture_data = {}
    for participant_id in range(1, 7):
        folder = f'rows_deleted/p{participant_id}'
        if not os.path.exists(folder): continue
        files = glob.glob(f'{folder}/*.npy')
        files = [f for f in files if not os.path.basename(f).startswith('rest')]
        for file in files:
            gesture_name = os.path.basename(file).split('_')[0]
            if gesture_name not in gesture_data:
                gesture_data[gesture_name] = []
            gesture_data[gesture_name].append(np.load(file))
    
    gen_model = GestureModel(window_size_samples=gesture_window_samples, sampling_rate=SAMPLINGHZ)
    gen_model.train(gesture_data)
    gen_model.save_model('gesture_model.pkl')
    print("✓ General Models saved.")
    return True

def Command(stream_buffer, stream_index, calib_buffer, calib_index, 
           recording_flag, recording_gesture, is_running_flag, Pis_running_flag, system): 
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
        case "ptr":
            print("Now running personal train function...")
            success = TrainPersonal()
            if success:
                system._load_models() # Modeli güncel halini belleğe al
        case "cb":  #calibrate
            if not system.is_data_acquisition_running():
                print("ERROR: Data acquisition not running. Use 'connect' first.")
            else:
                gesture_name = input("Which gesture would you like to calibrate? ")
                success = Calibrate(gesture_name, stream_buffer, stream_index, calib_buffer, calib_index, recording_flag, recording_gesture)
                if success:
                    print("\n Tip: Run 'tr' to retrain the personal model with new data")
        case "startcf":
            if not system.is_data_acquisition_running():
                print("ERROR: Data acquisition not running. Use 'connect' first.")
            else:
                print("starting general classification")
                is_running_flag.value = 1
                Pis_running_flag.value = 0
        case "stopcf":
            print("stopping all classification")
            is_running_flag.value = 0
            Pis_running_flag.value = 0
        case "startPcf":
            if not system.is_data_acquisition_running():
                print("ERROR: Data acquisition not running. Use 'connect' first.")
            else:
                print("starting personal classification")
                Pis_running_flag.value = 1
                is_running_flag.value = 0
        case "stopPcf":
            print("stopping all classification")
            is_running_flag.value = 0
            Pis_running_flag.value = 0
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
        self.Pis_running_flag = None
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
        self.Pis_running_flag = mp.Value('i', 0)
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
                sampling_rate=SAMPLINGHZ)
            if os.path.exists('gesture_model.pkl'):
                self.gesture_model.load_model('gesture_model.pkl')

            self.Pgesture_model = PersonalModel(
                window_size_samples=gesture_window_samples, 
                sampling_rate=SAMPLINGHZ)
            if os.path.exists('Pgesture_model.pkl'):
                self.gesture_model.load_model('Pgesture_model.pkl')
                
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
                self.Pis_running_flag,
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
    def start_personal_classification(self):
        """Enable personal classification (set Pis_running_flag to 1)"""
        self.is_running_flag.value = 0  # Stop general model
        self.Pis_running_flag.value = 1
        print("✓ Personal classification enabled")

    def stop_personal_classification(self):
        """Disable personal classification (set Pis_running_flag to 0)"""
        self.Pis_running_flag.value = 0
        print("✓ Personal classification disabled")

    def is_personal_classification_running(self):
        """Check if personal classification is active"""
        return bool(self.Pis_running_flag.value)
    
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
            success = Calibrate(
            gesture_name,
            self.stream_buffer,
            self.stream_index,
            self.calib_buffer, 
            self.calib_index,
            self.recording_flag, 
            self.recording_gesture)
            return success
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
                system.Pis_running_flag,
                system
            )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        system.cleanup()
        print("Goodbye!")
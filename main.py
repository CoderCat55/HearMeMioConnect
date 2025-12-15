"""handles multiprocessing and user commands"""
import multiprocessing as mp
from multiprocessing import Process, shared_memory
import numpy as np
from model import GestureClassifier
import time

# Constants
STREAM_BUFFER_SIZE = 1000  # ~5 seconds at 200Hz
CALIBRATION_BUFFER_SIZE = 600  # 3 seconds at 200Hz
###bufferların farklı boyutlarda olması sorun yaratır mı???


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
        
class GestureSystem:
    """Central system managing all components: shared memory, processes, and classifier"""
    
    def __init__(self):
        print("=== Initializing Gesture Recognition System ===")
        
        # Create shared memory buffers
        self.shm_stream = shared_memory.SharedMemory(create=True, size=STREAM_BUFFER_SIZE*34*4)
        self.stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 34), dtype=np.float32, buffer=self.shm_stream.buf)
        self.stream_buffer.fill(0)  # Initialize to zero
        
        self.shm_calib = shared_memory.SharedMemory(create=True, size=CALIBRATION_BUFFER_SIZE*34*4)
        self.calib_buffer = np.ndarray((CALIBRATION_BUFFER_SIZE, 34), dtype=np.float32, buffer=self.shm_calib.buf)
        self.calib_buffer.fill(0)
        
        # Create shared indices and flags
        self.stream_index = mp.Value('i', 0)
        self.calib_index = mp.Value('i', 0)
        self.recording_flag = mp.Value('i', 0)
        self.recording_gesture = mp.Array('c', 50)
        
        # Initialize classifier
        self.classifier = GestureClassifier()
        self.classifier.load_calibration_data()
        
        # Process reference
        self.data_process = None
        
        print("System initialized!")
    
    def start_data_acquisition(self):
        """Start the data acquisition process (called from /connect endpoint)"""
        if self.data_process is not None and self.data_process.is_alive():
            print("Data acquisition already running!")
            return False
        
        print("Starting data acquisition process...")
        self.data_process = Process(
            target=data_acquisition_process,
            args=(self.shm_stream.name, self.shm_calib.name, self.stream_index, 
                  self.calib_index, self.recording_flag, self.recording_gesture)
        )
        self.data_process.daemon = True  # Dies when main process dies
        self.data_process.start()
        
        print("Waiting for Myo connections...")
        time.sleep(5)  # Give it time to connect
        
        if not self.data_process.is_alive():
            print("ERROR: Data acquisition process failed to start!")
            return False
        
        print("Data acquisition process started successfully!")
        return True
    
    def stop_data_acquisition(self):
        """Stop the data acquisition process"""
        if self.data_process is not None and self.data_process.is_alive():
            print("Stopping data acquisition...")
            self.data_process.terminate()
            self.data_process.join(timeout=5)
            self.data_process = None
            print("Data acquisition stopped.")
    
    def is_data_acquisition_running(self):
        """Check if data acquisition is running"""
        return self.data_process is not None and self.data_process.is_alive()
    
    def calibrate(self, gesture_name):
        """Calibrate a gesture"""
        if not self.is_data_acquisition_running():
            print("ERROR: Data acquisition not running!")
            return False
        
        # Reset calibration buffer
        self.calib_index.value = 0
        
        # Set flag in shared memory
        gesture_bytes = gesture_name.encode('utf-8')
        for i, byte in enumerate(gesture_bytes[:50]):  # Max 50 chars
            self.recording_gesture[i] = byte
        self.recording_flag.value = 1  # Start recording
        
        # Wait 3 seconds
        print("Recording... ", end='', flush=True)
        for i in range(CALIBRATION_DURATION):
            time.sleep(1)
            print(f"{CALIBRATION_DURATION-i}... ", end='', flush=True)
        print("Done!")
        
        # Stop recording
        self.recording_flag.value = 0
        time.sleep(0.5)  # Let final batch flush
        
        # Read recorded data
        recorded_data = get_calibration_buffer_from_shared_mem(self.calib_buffer, self.calib_index)
        
        if recorded_data is None or len(recorded_data) == 0:
            print("ERROR: No data was recorded! Check if Myos are connected.")
            return False
        
        # Add to classifier
        self.classifier.add_calibration_sample(gesture_name, recorded_data)
        
        # Save to disk
        import os
        os.makedirs('calibration_data', exist_ok=True)
        timestamp = int(time.time())
        np.save(f'calibration_data/{gesture_name}_{timestamp}.npy', recorded_data)
        
        print(f"Calibration complete! Saved {len(recorded_data)} samples")
        return True
    
    def classify(self):
        """Classify current gesture"""
        if not self.is_data_acquisition_running():
            print("ERROR: Data acquisition not running!")
            return None
        
        time.sleep(CLASSIFICATION_DURATION)
        
        # Read current data from shared memory
        current_data = get_recent_data_from_shared_mem(
            self.stream_buffer, self.stream_index, window_seconds=CLASSIFICATION_DURATION
        )
        
        if current_data is None or len(current_data) < 10:
            print("ERROR: Not enough data to classify!")
            return None
        
        # Extract features
        features = GestureClassifier.extract_features(current_data)
        
        # Classify
        result = self.classifier.classify(features)
        print(f"Predicted gesture: {result}")
        return result
    
    def train(self):
        """Train the classifier"""
        print("Training model...")
        success = self.classifier.train()
        if success:
            self.classifier.save_model('model.pkl')
            print("Training complete!")
            return True
        else:
            print("Training failed! Make sure you have calibration data.")
            return False
    
    def live_classify(self, num_classifications=20):
        """Continuously classify gestures"""
        if self.classifier.model is None:
            print("ERROR: Model not trained yet! Please train the model first using 'tr'.")
            return
        
        print(f"\n>>> Starting LIVE classification... classify {num_classifications} times <<<")
        for i in range(1, num_classifications + 1):
            print(f"\nClassification {i}/{num_classifications}:")
            self.classify()
            print("waiting 1 sec for next classification")
            time.sleep(CLASSIFICATION_DURATION)
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.stop_data_acquisition()
        
        self.shm_stream.close()
        self.shm_stream.unlink()
        self.shm_calib.close()
        self.shm_calib.unlink()
        
        print("Cleanup complete!")

def command_loop(system):
    """Interactive command loop for testing"""
    print("\nSystem ready! Available commands: tr=train, cf=classify, cb=calibrate, live=live classify")
    print()
    
    try:
        while True:
            value = input("Enter your command: ")
            
            match value:
                case "tr":  # train
                    print("Training model...")
                    system.train()
                    
                case "cf":  # classify
                    print(f"Classify will start in ", end='', flush=True)
                    for i in range(CLASSIFICATION_STARTS, 0, -1):
                        print(f"{i}... ", end='', flush=True)
                        time.sleep(1)
                    print("\nClassifying gesture...")
                    system.classify()
                    
                case "cb":  # calibrate
                    gesture_name = input("Which gesture would you like to calibrate? ")
                    print(f"Calibration will start in ", end='', flush=True)
                    for i in range(CALIBRATION_STARTS, 0, -1):
                        print(f"{i}... ", end='', flush=True)
                        time.sleep(1)
                    print("\n")
                    print(f"Recording calibration for '{gesture_name}' - '{CALIBRATION_DURATION} seconds...")
                    system.calibrate(gesture_name)
                    
                case "live":  # live classification
                    print(f"Classification will start in ", end='', flush=True)
                    for i in range(CLASSIFICATION_STARTS, 0, -1):
                        print(f"{i}... ", end='', flush=True)
                        time.sleep(1)
                    print("\n")
                    system.live_classify()
                    
                case _:
                    print("Invalid command! Use: tr, cf, cb, or live")
                    
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    # Create the system
    system = GestureSystem()
    
    # Import and inject into webserver
    from webserver import app, inject_system
    inject_system(system)
    
    print("\n" + "="*50)
    print("Starting Flask web server...")
    print("="*50)
    
    # Run Flask server
    try:
        app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        system.cleanup()

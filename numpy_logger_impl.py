"""
Numpy Logger Process
Collects EMG+IMU data and saves windowed training examples to .npz files
"""

import numpy as np
from collections import deque
import time
import queue

WINDOW_SIZE = 150  # Samples (0.75 seconds at 200Hz)
OVERLAP = 100      # Samples
SHIFT = WINDOW_SIZE - OVERLAP  # 50 samples = 0.25 seconds


def numpy_logger_process(emg_queue, recording_flag, current_label, shutdown_event):
    """
    Main function for Numpy Logger process
    
    Args:
        emg_queue: Queue receiving EMG+IMU data packets
        recording_flag: Shared boolean - True when recording
        current_label: Shared string - current gesture label
        shutdown_event: Event to signal shutdown
    """
    print("🟢 Numpy Logger: Started")
    
    # Buffers for each armband
    buffers = {
        0: deque(maxlen=WINDOW_SIZE),
        1: deque(maxlen=WINDOW_SIZE)
    }
    
    # Storage for complete windows during recording
    recorded_windows = []
    recorded_labels = []
    was_recording = False
    
    while not shutdown_event.is_set():
        try:
            # Blocking read with timeout
            data = emg_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️  Numpy Logger: Error reading queue: {e}")
            continue
        
        armband_id = data['armband_id']
        
        # Combine EMG + IMU into single vector [17 values]
        # EMG: 8 values (channels 0-7)
        # IMU accel: 3 values (x, y, z)
        # IMU gyro: 3 values (x, y, z)
        # IMU orient: 3 values (x, y, z of quaternion, skipping w)
        try:
            combined = (
                data['emg'] + 
                data['imu']['accel'] + 
                data['imu']['gyro'] + 
                data['imu']['orientation'][1:]  # Skip quaternion w, keep x,y,z
            )
        except Exception as e:
            print(f"⚠️  Numpy Logger: Error combining data: {e}")
            continue
        
        if len(combined) != 17:
            print(f"⚠️  Numpy Logger: Expected 17 values, got {len(combined)}")
            continue
        
        # Add to buffer
        buffers[armband_id].append(combined)
        
        # Check if recording and both buffers are full
        if recording_flag.value and \
           len(buffers[0]) == WINDOW_SIZE and \
           len(buffers[1]) == WINDOW_SIZE:
            
            # Create window: shape (150, 2, 17)
            # Dim 0: time samples
            # Dim 1: armband ID
            # Dim 2: sensor channels (8 EMG + 9 IMU)
            window = np.zeros((WINDOW_SIZE, 2, 17))
            window[:, 0, :] = np.array(buffers[0])
            window[:, 1, :] = np.array(buffers[1])
            
            # Store
            recorded_windows.append(window)
            label = current_label.value.decode('utf-8').strip('\x00')
            recorded_labels.append(label)
            
            if len(recorded_windows) % 10 == 0:
                print(f"📊 Numpy Logger: Recorded {len(recorded_windows)} windows for '{label}'")
            
            # Shift buffers (overlapping windows for better training data)
            for _ in range(SHIFT):
                buffers[0].popleft()
                buffers[1].popleft()
            
            was_recording = True
        
        # Detect when recording stops - save file
        if was_recording and not recording_flag.value:
            save_recording(recorded_windows, recorded_labels)
            
            # Clear everything
            recorded_windows = []
            recorded_labels = []
            buffers[0].clear()
            buffers[1].clear()
            was_recording = False
    
    # Final save if recording when shutdown
    if len(recorded_windows) > 0:
        save_recording(recorded_windows, recorded_labels)
    
    print("🔴 Numpy Logger: Stopped")


def save_recording(windows, labels):
    """Save recorded windows to .npz file"""
    if len(windows) == 0:
        print("⚠️  Numpy Logger: No data to save")
        return
    
    gesture = labels[0]  # All labels should be the same
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"training_{gesture}_{timestamp}.npz"
    
    windows_array = np.array(windows)
    labels_array = np.array(labels)
    
    # Save with compression
    np.savez_compressed(
        filename,
        windows=windows_array,
        labels=labels_array,
        metadata={
            'gesture': gesture,
            'recording_date': timestamp,
            'sample_rate': 200,
            'window_size': WINDOW_SIZE,
            'overlap': OVERLAP,
            'num_examples': len(windows)
        }
    )
    
    print(f"💾 Numpy Logger: Saved {len(windows)} windows to {filename}")
    print(f"   Shape: {windows_array.shape}")
    print(f"   Size: {windows_array.nbytes / 1024:.1f} KB")

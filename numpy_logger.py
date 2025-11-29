import numpy as np
import time
from collections import deque
import os

class NumpyLogger:
    def __init__(self, emg_queue, recording_flag, current_label):
        self.emg_queue = emg_queue
        self.recording_flag = recording_flag
        self.current_label = current_label
        
        # Buffers for current recording session
        self.emg_buffer = deque(maxlen=1500)  # 7.5 seconds at 200Hz
        self.imu_buffer = deque(maxlen=1500)
        self.timestamps = deque(maxlen=1500)
        
        # Create data directory
        os.makedirs('training_data', exist_ok=True)
        
    def run(self):
        """Main logging loop"""
        print("Numpy Logger started...")
        
        while True:
            try:
                # Get data from queue (non-blocking)
                if not self.emg_queue.empty():
                    data_packet = self.emg_queue.get_nowait()
                    
                    # If recording is active, save to buffers
                    if self.recording_flag.value:
                        self._add_to_buffers(data_packet)
                        
            except:
                time.sleep(0.001)  # Small sleep to prevent CPU overload
    
    def _add_to_buffers(self, data_packet):
        """Add data to appropriate buffers"""
        if data_packet['type'] == 'emg':
            self.emg_buffer.append({
                'timestamp': data_packet['timestamp'],
                'device_name': data_packet['device_name'],
                'data': data_packet['data'],
                'sample_number': data_packet['sample_number']
            })
        elif data_packet['type'] == 'imu':
            self.imu_buffer.append(data_packet)
            
        # Check if we should save (when recording stops)
        if not self.recording_flag.value and len(self.emg_buffer) > 0:
            self._save_training_data()
    
    def _save_training_data(self):
        """Save buffered data to .npz file"""
        if len(self.emg_buffer) == 0:
            return
            
        gesture_label = self.current_label.value.decode().strip()
        timestamp = int(time.time())
        filename = f"training_data/{gesture_label}_{timestamp}.npz"
        
        # Convert buffers to numpy arrays
        emg_data = []
        imu_data = []
        
        for item in self.emg_buffer:
            emg_data.append(item['data'])
            
        for item in self.imu_buffer:
            imu_data.append(item)
        
        # Save to compressed numpy format
        np.savez_compressed(
            filename,
            emg_data=np.array(emg_data),
            imu_data=np.array(imu_data),
            gesture_label=gesture_label,
            timestamp=timestamp
        )
        
        print(f"Saved training data: {filename} with {len(emg_data)} samples")
        
        # Clear buffers
        self.emg_buffer.clear()
        self.imu_buffer.clear()

import struct
import math
import time

class SharedMemoryHandler:
    def __init__(self, myo_driver):
        # Reference to MyoDriver to access Myo objects and device names
        self.myo_driver = myo_driver
        self.device_buffers = {}  # key: device_name, value: shared memory buffer
    
    def _get_device_name(self, connection_id):
        """Get device name from connection ID"""
        for myo in self.myo_driver.myos:
            if myo.connection_id == connection_id:
                return myo.device_name
        return None
    
    def handle_emg(self, payload):
        connection_id = payload['connection']
        
        # Get device name from MyoDriver
        device_name = self._get_device_name(connection_id)
        if device_name is None:
            return  # Myo info not available yet
        
        # Parse the EMG data
        sample1 = struct.unpack('<8b', payload['value'][0:8])
        sample2 = struct.unpack('<8b', payload['value'][8:16])
        
        # Normalize to [-1, 1] range
        sample1_normalized = [x / 127.0 for x in sample1]
        sample2_normalized = [x / 127.0 for x in sample2]
        
        # Write to shared memory using device_name as key
        timestamp = time.time()
        self.write_to_shared_memory(device_name, sample1_normalized, timestamp)
        self.write_to_shared_memory(device_name, sample2_normalized, timestamp)
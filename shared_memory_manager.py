# File: shared_memory_manager.py
import numpy as np
from multiprocessing import shared_memory
import time

class SharedMemoryManager:
    """
    Manages circular buffers in shared memory for 2 Myo armbands.
    Each armband has separate EMG and IMU buffers.
    """
    def __init__(self, buffer_seconds=5, emg_rate=200, imu_rate=50):
        self.buffer_seconds = buffer_seconds
        self.emg_rate = emg_rate
        self.imu_rate = imu_rate
        
        # Calculate buffer sizes
        self.emg_buffer_size = buffer_seconds * emg_rate  # e.g., 5*200 = 1000 samples
        self.imu_buffer_size = buffer_seconds * imu_rate  # e.g., 5*50 = 250 samples
        
        # Structure: {device_name: {'emg': {...}, 'imu': {...}}}
        self.devices = {}
        
    def create_device_buffers(self, device_name):
        """Create shared memory buffers for a specific device"""
        if device_name in self.devices:
            return  # Already created
        
        # EMG buffer: (samples, 8_channels)
        emg_shape = (self.emg_buffer_size, 8)
        emg_shm = shared_memory.SharedMemory(
            create=True, 
            size=np.prod(emg_shape) * np.dtype(np.float32).itemsize
        )
        emg_array = np.ndarray(emg_shape, dtype=np.float32, buffer=emg_shm.buf)
        emg_array.fill(0)
        
        # EMG timestamps
        emg_ts_shm = shared_memory.SharedMemory(
            create=True,
            size=self.emg_buffer_size * np.dtype(np.float64).itemsize
        )
        emg_ts_array = np.ndarray((self.emg_buffer_size,), dtype=np.float64, buffer=emg_ts_shm.buf)
        emg_ts_array.fill(0)
        
        # IMU buffer: (samples, 9) - quat(4) + accel(3) + gyro(3) = 10, but we'll use 9 (roll,pitch,yaw,accel,gyro)
        imu_shape = (self.imu_buffer_size, 9)
        imu_shm = shared_memory.SharedMemory(
            create=True,
            size=np.prod(imu_shape) * np.dtype(np.float32).itemsize
        )
        imu_array = np.ndarray(imu_shape, dtype=np.float32, buffer=imu_shm.buf)
        imu_array.fill(0)
        
        # IMU timestamps
        imu_ts_shm = shared_memory.SharedMemory(
            create=True,
            size=self.imu_buffer_size * np.dtype(np.float64).itemsize
        )
        imu_ts_array = np.ndarray((self.imu_buffer_size,), dtype=np.float64, buffer=imu_ts_shm.buf)
        imu_ts_array.fill(0)
        
        # Store references
        self.devices[device_name] = {
            'emg': {
                'shm': emg_shm,
                'array': emg_array,
                'ts_shm': emg_ts_shm,
                'ts_array': emg_ts_array,
                'write_index': 0
            },
            'imu': {
                'shm': imu_shm,
                'array': imu_array,
                'ts_shm': imu_ts_shm,
                'ts_array': imu_ts_array,
                'write_index': 0
            }
        }
        
        print(f"Created shared memory buffers for device: {device_name}")
    
    def write_emg(self, device_name, sample, timestamp):
        """Write single EMG sample to circular buffer"""
        if device_name not in self.devices:
            self.create_device_buffers(device_name)
        
        emg_data = self.devices[device_name]['emg']
        idx = emg_data['write_index'] % self.emg_buffer_size
        
        emg_data['array'][idx] = sample
        emg_data['ts_array'][idx] = timestamp
        emg_data['write_index'] += 1
    
    def write_imu(self, device_name, imu_values, timestamp):
        """Write single IMU sample to circular buffer"""
        if device_name not in self.devices:
            self.create_device_buffers(device_name)
        
        imu_data = self.devices[device_name]['imu']
        idx = imu_data['write_index'] % self.imu_buffer_size
        
        imu_data['array'][idx] = imu_values
        imu_data['ts_array'][idx] = timestamp
        imu_data['write_index'] += 1
    
    def cleanup(self):
        """Close and unlink all shared memory"""
        for device_name, buffers in self.devices.items():
            buffers['emg']['shm'].close()
            buffers['emg']['shm'].unlink()
            buffers['emg']['ts_shm'].close()
            buffers['emg']['ts_shm'].unlink()
            buffers['imu']['shm'].close()
            buffers['imu']['shm'].unlink()
            buffers['imu']['ts_shm'].close()
            buffers['imu']['ts_shm'].unlink()
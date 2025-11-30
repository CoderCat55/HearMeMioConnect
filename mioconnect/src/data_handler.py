import struct
import math
import time
import numpy as np

class DataHandler:
    def __init__(self, config, stream_buffer=None, stream_index=None, 
                 calib_buffer=None, calib_index=None, recording_flag=None,
                 recording_gesture=None):
        self.printEmg = config.PRINT_EMG
        self.printImu = config.PRINT_IMU
        
        # Streaming buffer (circular)
        self.stream_buffer = stream_buffer
        self.stream_index = stream_index
        
        # Calibration buffer (linear, fixed size)
        self.calib_buffer = calib_buffer
        self.calib_index = calib_index
        self.recording_flag = recording_flag
        self.recording_gesture = recording_gesture
        
        # Latest values from each Myo (for real-time classification)
        self.myo1_latest = np.zeros(17, dtype=np.float32)  # 8 EMG + 9 IMU
        self.myo2_latest = np.zeros(17, dtype=np.float32)
        
        # Batch accumulation (write to shared memory in batches)
        self.local_buffer = []
        self.BATCH_SIZE = 20
        
    def handle_emg(self, payload, myo_driver):
        """Handle EMG data - Store RAW values (no normalization)"""
        connection_id = payload['connection']
        device_name = myo_driver.get_device_name(connection_id)
        
        if device_name is None:
            return  # Myo info not available yet
        
        if self.printEmg:
            print(f"EMG from {device_name}: {payload['value']}")
        
        # Extract BOTH samples (don't waste data!)
        sample1 = struct.unpack('<8b', payload['value'][0:8])   # First sample
        sample2 = struct.unpack('<8b', payload['value'][8:16])  # Second sample
        
        timestamp = time.time()
        
        # Process both samples
        for sample in [sample1, sample2]:
            self._process_single_emg_sample(sample, device_name, timestamp)
    
    def _process_single_emg_sample(self, emg_data, device_name, timestamp):
        """Process a single EMG sample (8 channels)"""
        # Store in latest values (for real-time classification)
        if device_name == "Myo-0" or device_name.endswith("..."):  # Adjust based on actual names
            self.myo1_latest[0:8] = emg_data
        else:
            self.myo2_latest[0:8] = emg_data
        
        # For now, accumulate in local buffer
        # TODO: Combine with IMU data before writing to shared memory
        # (Currently we're storing EMG and IMU separately, which causes sync issues)
    
    def handle_imu(self, payload, myo_driver):
        """Handle IMU data"""
        connection_id = payload['connection']
        device_name = myo_driver.get_device_name(connection_id)
        
        if device_name is None:
            return
        
        if self.printImu:
            print(f"IMU from {device_name}")
        
        # Parse orientation (quaternion)
        data = payload['value'][0:8]
        w, x, y, z = struct.unpack('hhhh', data)
        roll, pitch, yaw = self._euler_angle(w, x, y, z)
        
        # Parse accelerometer
        accel_data = payload['value'][8:14]
        ax, ay, az = struct.unpack('hhh', accel_data)
        
        # Parse gyroscope
        gyro_data = payload['value'][14:20]
        gx, gy, gz = struct.unpack('hhh', gyro_data)
        
        timestamp = time.time()
        
        # Store in latest values
        imu_values = np.array([roll, pitch, yaw, ax, ay, az, gx, gy, gz], dtype=np.float32)
        
        if device_name == "Myo-0":  # Adjust based on actual names
            self.myo1_latest[8:17] = imu_values
        else:
            self.myo2_latest[8:17] = imu_values
        
        # Combine data from both Myos and write to buffer
        self._write_combined_sample(timestamp)
    
    def _write_combined_sample(self, timestamp):
        """Combine data from both Myos and write to shared memory"""
        if self.stream_buffer is None:
            return
        
        # Combined sample: [myo1_emg(8), myo1_imu(9), myo2_emg(8), myo2_imu(9)] = 34 features
        combined = np.concatenate([self.myo1_latest, self.myo2_latest])
        
        # Accumulate in local buffer
        self.local_buffer.append((timestamp, combined))
        
        # Write in batches
        if len(self.local_buffer) >= self.BATCH_SIZE:
            self._flush_to_shared_memory()
    
    def _flush_to_shared_memory(self):
        """Write accumulated samples to shared memory"""
        if len(self.local_buffer) == 0:
            return
        
        # Write to streaming buffer (circular)
        for timestamp, sample in self.local_buffer:
            idx = self.stream_index.value % len(self.stream_buffer)
            self.stream_buffer[idx] = sample
            self.stream_index.value += 1
            
            # If recording, also write to calibration buffer
            if self.recording_flag.value == 1 and self.calib_index.value < len(self.calib_buffer):
                self.calib_buffer[self.calib_index.value] = sample
                self.calib_index.value += 1
        
        self.local_buffer.clear()

    @staticmethod
    def _euler_angle(w, x, y, z):
        """Convert quaternion to Euler angles"""
        # roll (x-axis rotation)
        sinr_cosp = +2.0 * (w * x + y * z)
        cosr_cosp = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = +2.0 * (w * y - z * x)
        if math.fabs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = +2.0 * (w * z + x * y)
        cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
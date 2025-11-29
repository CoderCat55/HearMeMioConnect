import struct
import math
import time
from multiprocessing import Queue

class DataHandler:
    """
    EMG/IMU data handler for multiprocessing framework.
    Modified to send data to queues instead of OSC.
    """
    def __init__(self, config, emg_queue_logger, emg_queue_classifier, myo_driver):
        self.printEmg = config.PRINT_EMG
        self.printImu = config.PRINT_IMU
        
        # Multiprocessing queues for data sharing
        self.emg_queue_logger = emg_queue_logger
        self.emg_queue_classifier = emg_queue_classifier
        
        # Reference to MyoDriver to access Myo objects and device names
        self.myo_driver = myo_driver

        # Reference to MyoDriver to access Myo objects and device names
        self.myo_driver = myo_driver
        
        # ADD THIS: Buffer to hold latest EMG samples
        self.latest_emg_samples = {}  # {device_name: [(sample1, timestamp), (sample2, timestamp)]}
        
        # ADD THIS: Device name to armband ID mapping
        self.device_name_to_id = {}

    def handle_emg(self, payload):
        """
        Handle EMG data and send to multiprocessing queues.
        Based on original _send_single_emg method.
        """
        connection_id = payload['connection']
        raw_data = payload['value']
        
        # Get device name from MyoDriver
        device_name = self._get_device_name(connection_id)
        if device_name is None:
            return  # Myo info not available yet
        
        # Parse EMG data (2 samples of 8 channels each) - SAME AS ORIGINAL
        sample1 = struct.unpack('<8b', raw_data[0:8])   # First sample: 8 signed bytes
        sample2 = struct.unpack('<8b', raw_data[8:16])  # Second sample: 8 signed bytes
        
        timestamp = time.time()
        # Store both samples with timestamps
        self.latest_emg_samples[device_name] = [
            (sample1, timestamp),
            (sample2, timestamp + 0.005)  # 5ms later (200Hz)
        ]

        if self.printEmg:
            print(f"EMG {device_name}: Sample1={sample1[:3]}..., Sample2={sample2[:3]}...")


    def handle_imu(self, payload):
        """
        Handle IMU data and send to multiprocessing queues.
        Based on original handle_imu method.
        """
        connection_id = payload['connection']
        raw_data = payload['value']
        
        # Get device name from MyoDriver
        device_name = self._get_device_name(connection_id)
        if device_name is None:
            return  # Myo info not available yet
        
        timestamp = time.time()
        

        orientation_data = raw_data[0:8]
        w, x, y, z = struct.unpack('hhhh', orientation_data)  # 4 signed shorts
        roll, pitch, yaw = self._euler_angle(w, x, y, z)
        
        
        accel_data = raw_data[8:14]
        accel_x, accel_y, accel_z = struct.unpack('hhh', accel_data)  # 3 signed shorts
        
        
        gyro_data = raw_data[14:20]
        gyro_x, gyro_y, gyro_z = struct.unpack('hhh', gyro_data)  # 3 signed shorts
        # Check if we have EMG samples for this device
        if device_name not in self.latest_emg_samples:
            return  # No EMG yet, skip
        
        # Get the 2 most recent EMG samples
        emg_samples = self.latest_emg_samples[device_name]
        
        # Send COMBINED packets for BOTH EMG samples
        for emg_sample, emg_timestamp in emg_samples:
            combined_packet = {
                'timestamp': emg_timestamp,
                'armband_id': self._get_armband_id(device_name),
                'emg': emg_sample,  # 8 values
                'imu': {
                    'accel': [accel_x, accel_y, accel_z],
                    'gyro': [gyro_x, gyro_y, gyro_z],
                    'orientation': [w, x, y, z]
                }
            }
            
            try:
                self.emg_queue_logger.put(combined_packet, block=False)
                self.emg_queue_classifier.put(combined_packet, block=False)
            except:
                pass
        
        if self.printImu:
            print(f"IMU {device_name}: Roll={roll/math.pi:.3f}, Pitch={pitch/math.pi:.3f}, Yaw={yaw/math.pi:.3f}")
    
    def _get_device_name(self, connection_id):
        """
        Get device name from MyoDriver using connection ID.
        Returns None if Myo info not available yet.
        """
        for myo in self.myo_driver.myos:
            if myo.connection_id == connection_id and myo.device_name is not None:
                return myo.device_name
        return None
    
    def _get_armband_id(self, device_name):
        """
        Map device name to armband ID (0 or 1).
        First device connected = 0, second = 1
        """
        if device_name not in self.device_name_to_id:
            self.device_name_to_id[device_name] = len(self.device_name_to_id)
            print(f"📱 Armband '{device_name}' assigned ID: {self.device_name_to_id[device_name]}")
        
        return self.device_name_to_id[device_name]

    @staticmethod
    def _euler_angle(w, x, y, z):
        """
        Convert quaternion to Euler angles.
        EXACTLY THE SAME AS ORIGINAL - from Wikipedia.
        """
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

    @staticmethod
    def _vector_magnitude(x, y, z):
        return math.sqrt(x * x + y * y + z * z)
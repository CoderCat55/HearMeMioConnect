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
        
        # Create data packets for both samples - NORMALIZED like original
        data_packet1 = {
            'type': 'emg',
            'timestamp': timestamp,
            'device_name': device_name,
            'connection_id': connection_id,
            'data': sample1,  #raw filtered values
            'sample_number': 0,
            'raw_data': sample1  # Keep raw values for reference
        }
        
        data_packet2 = {
            'type': 'emg', 
            'timestamp': timestamp + 0.005,  # ~5ms between samples at 200Hz
            'device_name': device_name,
            'connection_id': connection_id,
            'data': [x / 127.0 for x in sample2],  # Normalize to [-1, 1]
            'sample_number': 1,
            'raw_data': sample2
        }
        
        # Send to both queues
        try:
            self.emg_queue_logger.put(data_packet1)
            self.emg_queue_classifier.put(data_packet1)
            self.emg_queue_logger.put(data_packet2)
            self.emg_queue_classifier.put(data_packet2)
        except:
            pass  # Handle queue full situations
        
        if self.printEmg:
            print(f"EMG {device_name}: Sample1={sample1}, Normalized={[x/127.0 for x in sample1]}")

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
        
        # Parse orientation (quaternion) - SAME AS ORIGINAL
        orientation_data = raw_data[0:8]
        w, x, y, z = struct.unpack('hhhh', orientation_data)  # 4 signed shorts
        roll, pitch, yaw = self._euler_angle(w, x, y, z)
        
        # Parse accelerometer - SAME AS ORIGINAL
        accel_data = raw_data[8:14]
        accel_x, accel_y, accel_z = struct.unpack('hhh', accel_data)  # 3 signed shorts
        
        # Parse gyroscope - SAME AS ORIGINAL  
        gyro_data = raw_data[14:20]
        gyro_x, gyro_y, gyro_z = struct.unpack('hhh', gyro_data)  # 3 signed shorts
        
        # Create IMU data packet - NORMALIZED like original
        imu_packet = {
            'type': 'imu',
            'timestamp': timestamp,
            'device_name': device_name,
            'connection_id': connection_id,
            'orientation': {
                'roll': roll / math.pi,        # Normalized to [-1, 1] like original
                'pitch': pitch / math.pi,      # Normalized to [-1, 1]  
                'yaw': yaw / math.pi,          # Normalized to [-1, 1]
                'quaternion': [w, x, y, z]     # Raw quaternion values
            },
            'accelerometer': {
                'x': accel_x,
                'y': accel_y, 
                'z': accel_z,
                'magnitude': self._vector_magnitude(accel_x, accel_y, accel_z)
            },
            'gyroscope': {
                'x': gyro_x,
                'y': gyro_y,
                'z': gyro_z,
                'magnitude': self._vector_magnitude(gyro_x, gyro_y, gyro_z)
            }
        }
        
        # Send to both queues
        try:
            self.emg_queue_logger.put(imu_packet)
            self.emg_queue_classifier.put(imu_packet)
        except:
            pass
        
        if self.printImu:
            print(f"IMU {device_name}: "
                  f"Roll={roll/math.pi:.3f}, Pitch={pitch/math.pi:.3f}, Yaw={yaw/math.pi:.3f}")

    def _get_device_name(self, connection_id):
        """
        Get device name from MyoDriver using connection ID.
        Returns None if Myo info not available yet.
        """
        for myo in self.myo_driver.myos:
            if myo.connection_id == connection_id and myo.device_name is not None:
                return myo.device_name
        return None

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
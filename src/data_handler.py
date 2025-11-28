from pythonosc import udp_client
import struct
import math
import time


class DataHandler:
    """
    Optimized EMG/IMU data handler with batched OSC messages and synchronized CSV logging.
    
    CRITICAL OPTIMIZATION: Changed from sending 2 separate EMG OSC messages per packet
    to sending 1 batched message with both samples. This reduces OSC overhead by 50%.
    """
    
    def __init__(self, config, csv_logger=None):
        """
        Initialize data handler with optional CSV logging
        
        Args:
            config: Configuration object with OSC settings and print flags
            csv_logger: Optional CSVLogger instance for synchronized data persistence
            
        REASONING: csv_logger now uses update_emg() and update_imu() instead of
        queuing individual samples. Timer thread handles actual CSV writing.
        """
        self.osc = udp_client.SimpleUDPClient(config.OSC_ADDRESS, config.OSC_PORT)
        self.printEmg = config.PRINT_EMG
        self.printImu = config.PRINT_IMU
        self.csv_logger = csv_logger

    def handle_emg(self, payload):
        """
        Handle EMG data with optimized OSC batching and CSV updates
        
        Args:
            payload: Dict with 'connection', 'atthandle', 'value' keys
                     value is 16 bytes: 2 samples x 8 channels x 1 byte
                     
        REASONING: From myodriver.py:handle_attribute_value(), EMG data arrives as
        ble_evt_attclient_attribute_value events with 2 samples per packet.
        
        CRITICAL OPTIMIZATION: Batches both samples into ONE OSC message.
        """
        if self.printEmg:
            print("EMG", payload['connection'], payload['atthandle'], payload['value'])

        # Unpack both EMG samples from the packet
        # REASONING: EMG packet structure from Myo protocol (myohw.py):
        # - Bytes 0-7: Sample 1 (channels 0-7)
        # - Bytes 8-15: Sample 2 (channels 0-7)
        emg_sample1 = struct.unpack('<8b', payload['value'][0:8])
        emg_sample2 = struct.unpack('<8b', payload['value'][8:16])
        
        # OPTIMIZATION: Send both samples in ONE OSC message instead of two
        # This reduces network overhead and improves latency for gesture recognition
        self._send_batched_emg(payload['connection'], emg_sample1, emg_sample2)
        
        # Update CSV logger with latest values (non-blocking)
        # REASONING: Timer thread will sample these values at fixed intervals
        if self.csv_logger:
            self.csv_logger.update_emg(payload['connection'], emg_sample1, emg_sample2)

    def _send_batched_emg(self, conn, sample1, sample2):
        """
        Send both EMG samples in a single OSC message (OPTIMIZED)
        
        Args:
            conn: Connection ID
            sample1: Tuple of 8 EMG values (first sample, channels 0-7)
            sample2: Tuple of 8 EMG values (second sample, channels 0-7)
            
        REASONING: Original code sent 2 separate messages. By batching, we reduce:
        - OSC message overhead (headers, network packets)
        - UDP socket operations (send() calls)
        - Processing time on receiver side
        
        Message format: /myo/emg [connection_id] [sample1_ch0-7] [sample2_ch0-7]
        Total: 1 string + 16 floats = 17 arguments instead of 2x9 arguments
        """
        builder = udp_client.OscMessageBuilder("/myo/emg")
        builder.add_arg(str(conn), 's')
        
        # Add first sample (channels 0-7)
        for value in sample1:
            builder.add_arg(value / 127.0, 'f')  # Normalize to [-1, 1]
        
        # Add second sample (channels 0-7)
        for value in sample2:
            builder.add_arg(value / 127.0, 'f')  # Normalize to [-1, 1]
        
        self.osc.send(builder.build())

    def handle_imu(self, payload):
        """
        Handle IMU data with CSV updates
        
        Args:
            payload: Dict with 'connection', 'atthandle', 'value' keys
                     value is 20 bytes:
                     - Bytes 0-7: Quaternion (4 x int16)
                     - Bytes 8-13: Accelerometer (3 x int16)
                     - Bytes 14-19: Gyroscope (3 x int16)
                     
        REASONING: From myodriver.py:handle_attribute_value(), IMU data arrives from
        ServiceHandles.IMUDataCharacteristic. Parse and update CSV logger.
        """
        if self.printImu:
            print("IMU", payload['connection'], payload['atthandle'], payload['value'])
        
        # Parse IMU data structure (from Myo Bluetooth protocol)
        # Quaternion for orientation
        quat = struct.unpack('hhhh', payload['value'][0:8])
        roll, pitch, yaw = self._euler_angle(*quat)
        
        # Accelerometer (raw values)
        accel = struct.unpack('hhh', payload['value'][8:14])
        
        # Gyroscope (raw values)
        gyro = struct.unpack('hhh', payload['value'][14:20])
        
        # Send OSC messages (original behavior maintained)
        # Orientation
        builder = udp_client.OscMessageBuilder("/myo/orientation")
        builder.add_arg(str(payload['connection']), 's')
        builder.add_arg(roll / math.pi, 'f')   # Normalize to [-1, 1]
        builder.add_arg(pitch / math.pi, 'f')
        builder.add_arg(yaw / math.pi, 'f')
        self.osc.send(builder.build())

        # Accelerometer (magnitude)
        builder = udp_client.OscMessageBuilder("/myo/accel")
        builder.add_arg(str(payload['connection']), 's')
        builder.add_arg(self._vector_magnitude(*accel), 'f')
        self.osc.send(builder.build())

        # Gyroscope (magnitude)
        builder = udp_client.OscMessageBuilder("/myo/gyro")
        builder.add_arg(str(payload['connection']), 's')
        builder.add_arg(self._vector_magnitude(*gyro), 'f')
        self.osc.send(builder.build())

        # Update CSV logger with latest IMU values (non-blocking)
        # REASONING: Timer thread will sample these values at fixed intervals
        if self.csv_logger:
            imu_data = {
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'accel_x': accel[0],
                'accel_y': accel[1],
                'accel_z': accel[2],
                'gyro_x': gyro[0],
                'gyro_y': gyro[1],
                'gyro_z': gyro[2]
            }
            self.csv_logger.update_imu(payload['connection'], imu_data)

    @staticmethod
    def _euler_angle(w, x, y, z):
        """
        Convert quaternion to Euler angles
        
        REASONING: From https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        (referenced in original code)
        """
        # roll (x-axis rotation)
        sinr_cosp = +2.0 * (w * x + y * z)
        cosr_cosp = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = +2.0 * (w * y - z * x)
        if math.fabs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = +2.0 * (w * z + x * y)
        cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _vector_magnitude(x, y, z):
        """Calculate 3D vector magnitude"""
        return math.sqrt(x * x + y * y + z * z)
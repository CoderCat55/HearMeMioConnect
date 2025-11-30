from pythonosc import udp_client
import struct
import math
import time

"""we need to delete the osc
   probobaly also need to add time for data
   also use device name not connection ID
"""
"""
Here is how to get device name. I dont want to use conenction ID because armbands can connect at different orders.

  def _get_device_name(self, connection_id):
        #Get device name from connection ID
        for myo in self.myo_driver.myos:
            if myo.connection_id == connection_id:
                return myo.device_name
        return None

# Get device name from MyoDriver
device_name = self._get_device_name(connection_id)
    if device_name is None:
        return  # Myo info not available yet
"""

class DataHandler:
    """
    EMG/IMU/Classifier data handler.
    """
    """def __init__(self, config):
        self.printEmg = config.PRINT_EMG
        self.printImu = config.PRINT_IMU"""
    def __init__(self, config, shared_buffer, buffer_index):
        self.printEmg = config.PRINT_EMG
        self.printImu = config.PRINT_IMU
        self.shared_buffer = shared_buffer  # numpy array in shared memory
        self.buffer_index = buffer_index  # shared integer
        self.local_buffer = []  # Accumulate here
        self.BATCH_SIZE = 20  # Write every 20 samples

    def _flush_to_shared_memory(self):
        """Write accumulated samples to shared memory"""
        if len(self.local_buffer) > 0:
            # Write batch to shared memory
            # Update buffer_index
            self.local_buffer.clear()

    def handle_emg(self, payload,myo_driver):
        """
        Handle EMG data.
        :param payload: emg data as two samples in a single pack.
        """
        connection_id = payload['connection']
        device_name = myo_driver.get_device_name(connection_id)
        if self.printEmg:
            print("EMG", payload['connection'], payload['atthandle'], payload['value'])

        # Send both samples
        self._send_single_emg(payload['connection'], payload['value'][0:8]) #use only this
        self._send_single_emg(payload['connection'], payload['value'][8:16])
        
    def handle_imu(self, payload,myo_driver):
        """
        Handle IMU data.
        :param payload: imu data in a single byte array.
        """
        connection_id = payload['connection']
        device_name = myo_driver.get_device_name(connection_id)
        if self.printEmg:
            print("EMG", payload['connection'], payload['atthandle'], payload['value'])

        if self.printImu:
            print("IMU", payload['connection'], payload['atthandle'], payload['value'])
        # Send orientation
        data = payload['value'][0:8]
        #builder = udp_client.OscMessageBuilder("/myo/orientation")
        #builder.add_arg(str(payload['connection']), 's')
        roll, pitch, yaw = self._euler_angle(*(struct.unpack('hhhh', data)))
        # Normalize to [-1, 1]
        #builder.add_arg(roll / math.pi, 'f')
        #builder.add_arg(pitch / math.pi, 'f')
        #builder.add_arg(yaw / math.pi, 'f')
     

        # Send accelerometer
        data = payload['value'][8:14]
        #builder = udp_client.OscMessageBuilder("/myo/accel")
        #builder.add_arg(str(payload['connection']), 's')
        #builder.add_arg(self._vector_magnitude(*(struct.unpack('hhh', data))), 'f')
     

        # Send gyroscope
        data = payload['value'][14:20]
        #builder = udp_client.OscMessageBuilder("/myo/gyro")
        #builder.add_arg(str(payload['connection']), 's')
        #builder.add_arg(self._vector_magnitude(*(struct.unpack('hhh', data))), 'f')
      

    @staticmethod
    def _euler_angle(w, x, y, z):
        """
        From https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles.
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

    @staticmethod #do we need this ??
    def _vector_magnitude(x, y, z):
        return math.sqrt(x * x + y * y + z * z)
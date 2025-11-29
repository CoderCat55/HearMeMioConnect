"""
Myo Handler Process
Connects to 2 Myo armbands and streams EMG+IMU data to queues

NOTE: This is a TEMPLATE showing how to integrate MioConnect.
You need to modify MioConnect's data_handler.py to use this approach.
Yes I have made some changes but please check them
"""

import time
import queue


from src.myodriver import MyoDriver
from src.config import Config

def myo_handler_process(emg_queue_logger, emg_queue_classifier, shutdown_event):
    """
    Real Myo Handler using MioConnect
    """
    print("🟢 Myo Handler: Started")
    
    from src.myodriver import MyoDriver
    from src.config import Config
    
    try:
        # Create config
        config = Config()
        config.MYO_AMOUNT = 2
        config.EMG_MODE = EmgMode.myohw_emg_mode_send_emg_raw
        config.IMU_MODE = ImuMode.myohw_imu_mode_send_data
        config.PRINT_EMG = False
        config.PRINT_IMU = False
        
        # Create driver with queues
        driver = MyoDriver(config, emg_queue_logger, emg_queue_classifier)
        
        # Run connection
        driver.run()
        
        print("🔵 Myo Handler: Connected, receiving data...")
        
        # Keep receiving until shutdown
        while not shutdown_event.is_set():
            driver.receive()
            time.sleep(0.001)  # Small sleep to prevent 100% CPU
        
    except KeyboardInterrupt:
        print("🔴 Myo Handler: Interrupted")
    except Exception as e:
        print(f"🔴 Myo Handler: Error - {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔴 Myo Handler: Stopped")
        
class MyoDataHandler:
    """
    Custom data handler to be integrated into MioConnect
    
    This replaces or extends MioConnect's DataHandler class
    """
    
    def __init__(self, queue_logger, queue_classifier):
        self.queue_logger = queue_logger
        self.queue_classifier = queue_classifier
        
        # Store latest EMG data temporarily (since EMG and IMU arrive separately)
        self.latest_emg = {0: None, 1: None}
        self.latest_timestamp = {0: 0, 1: 0}
        
        # Armband address to ID mapping
        self.address_to_id = {}
        self.next_id = 0
    
    def get_armband_id(self, address):
        """Convert Bluetooth address to armband ID (0 or 1)"""
        if address not in self.address_to_id:
            self.address_to_id[address] = self.next_id
            self.next_id += 1
            print(f"🔵 Myo Handler: Armband {self.address_to_id[address]} connected ({address})")
        return self.address_to_id[address]
    
    def handle_emg(self, address, emg_data):
        """
        Called by MioConnect when EMG data arrives
        
        Args:
            address: Bluetooth address of the armband
            emg_data: List of 8 EMG channel values (int16)
        """
        armband_id = self.get_armband_id(address)
        
        # Store EMG temporarily (waiting for IMU to arrive)
        self.latest_emg[armband_id] = list(emg_data)
        self.latest_timestamp[armband_id] = time.perf_counter()
    
    def handle_imu(self, address, imu_data):
        """
        Called by MioConnect when IMU data arrives
        
        Args:
            address: Bluetooth address of the armband
            imu_data: List of IMU values
        
        IMPORTANT: You need to verify the exact format from MioConnect!
        Typical format: [qw, qx, qy, qz, ax, ay, az, gx, gy, gz]
        - q: quaternion (orientation)
        - a: accelerometer
        - g: gyroscope
        """
        armband_id = self.get_armband_id(address)
        
        # TODO: Parse IMU data based on actual MioConnect format
        # This is a GUESS - verify with actual data!
        try:
            if len(imu_data) >= 10:
                orientation = imu_data[0:4]  # qw, qx, qy, qz
                accel = imu_data[4:7]        # ax, ay, az
                gyro = imu_data[7:10]        # gx, gy, gz
            else:
                print(f"⚠️  Myo Handler: Unexpected IMU data length: {len(imu_data)}")
                return
        except Exception as e:
            print(f"⚠️  Myo Handler: Error parsing IMU: {e}")
            return
        
        # Create complete data packet
        if self.latest_emg[armband_id] is not None:
            data_packet = {
                'timestamp': time.perf_counter(),
                'armband_id': armband_id,
                'emg': self.latest_emg[armband_id],
                'imu': {
                    'accel': list(accel),
                    'gyro': list(gyro),
                    'orientation': list(orientation)
                }
            }
            
            # Put in BOTH queues (non-blocking to avoid deadlock)
            try:
                self.queue_logger.put(data_packet, block=False)
            except queue.Full:
                pass  # Drop if full
            
            try:
                self.queue_classifier.put(data_packet, block=False)
            except queue.Full:
                pass  # Drop if full


# ============================================================================
# HOW TO INTEGRATE WITH MIOCONNECT
# ============================================================================

"""
STEP 1: Modify MioConnect's data_handler.py

In MioConnect/src/data_handler.py, find the DataHandler class and modify:

class DataHandler:
    def __init__(self, config_obj, queue_logger, queue_classifier):
        self.config = config_obj
        self.queue_logger = queue_logger      # ADD THIS
        self.queue_classifier = queue_classifier  # ADD THIS
        # ... rest of init

    def handle_emg(self, address, emg):
        # Instead of sending via OSC, put in queues
        armband_id = self.get_armband_id(address)
        self.latest_emg[armband_id] = emg
    
    def handle_imu(self, address, imu):
        # Combine with EMG and put in queues
        # (Follow MyoDataHandler.handle_imu logic above)


STEP 2: Modify MioConnect's main loop

In mio_connect.py or wherever the main loop is:

def myo_handler_process(queue_logger, queue_classifier, shutdown):
    config = Config()
    config.MYO_AMOUNT = 2
    
    # Pass queues to data handler
    data_handler = DataHandler(config, queue_logger, queue_classifier)
    
    driver = MyoDriver(config, data_handler)
    driver.connect()
    
    while not shutdown.is_set():
        driver.update()  # or whatever the main loop is
    
    driver.disconnect()


STEP 3: Verify IMU data format

Print IMU data to see actual structure:

def handle_imu(self, address, imu):
    print(f"IMU data ({len(imu)} values): {imu}")
    # Adjust parsing based on what you see
"""


# ============================================================================
# DEBUGGING: TEST WITHOUT MIOCONNECT
# ============================================================================

def myo_handler_mock(emg_queue_logger, emg_queue_classifier, shutdown_event):
    """
    Mock Myo Handler for testing without actual armbands
    Generates synthetic EMG+IMU data
    """
    print("🟡 Myo Handler (MOCK MODE): Started")
    print("⚠️  Using synthetic data for testing")
    
    import random
    
    while not shutdown_event.is_set():
        # Generate synthetic data at ~200Hz (5ms delay)
        for armband_id in [0, 1]:
            data_packet = {
                'timestamp': time.perf_counter(),
                'armband_id': armband_id,
                'emg': [random.randint(-128, 127) for _ in range(8)],
                'imu': {
                    'accel': [random.uniform(-2, 2) for _ in range(3)],
                    'gyro': [random.uniform(-200, 200) for _ in range(3)],
                    'orientation': [
                        random.uniform(-1, 1),  # qw
                        random.uniform(-1, 1),  # qx
                        random.uniform(-1, 1),  # qy
                        random.uniform(-1, 1)   # qz
                    ]
                }
            }
            
            try:
                emg_queue_logger.put(data_packet, block=False)
                emg_queue_classifier.put(data_packet, block=False)
            except queue.Full:
                pass
        
        time.sleep(0.005)  # 5ms = 200Hz
    
    print("🔴 Myo Handler (MOCK): Stopped")


# For testing, export the mock version
myo_handler_process = myo_handler_mock  # Use mock by default
# After integrating MioConnect, change back to real version

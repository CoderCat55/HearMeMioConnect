"""
CSV Logger with threading support for Myo data
Minimal impact on data collection
"""
import csv
import threading
import queue
import time
from datetime import datetime


class CSVLogger:
    """
    Asynchronous CSV logger that writes data in a separate thread
    to minimize impact on data collection.
    """
    
    def __init__(self, filename_prefix="myo_data"):
        """
        Initialize CSV logger
        
        Args:
            filename_prefix: Base name for CSV files (timestamp will be appended)
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{filename_prefix}_{timestamp}.csv"
        
        # Thread-safe queue for data
        self.data_queue = queue.Queue(maxsize=10000)  # Buffer up to 10k samples
        
        # Threading control
        self.stop_event = threading.Event()
        self.writer_thread = None
        
        # Device name mapping: connection_id -> device_name
        self.device_names = {}
        
        # CSV file handle
        self.csv_file = None
        self.csv_writer = None
        
    def register_device(self, connection_id, device_name):
        """
        Register a device name for a connection ID
        
        Args:
            connection_id: Connection ID (0, 1, 2...)
            device_name: Device name from Myo
        """
        self.device_names[connection_id] = device_name
        print(f"CSV Logger: Registered device '{device_name}' with connection {connection_id}")
        
    def start(self):
        """Start the CSV writer thread"""
        # Initialize CSV file
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow([
            'timestamp', 'device_name', 'connection_id', 
            'data_type', 'data_values'
        ])
        self.csv_file.flush()
        
        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        print(f"CSV Logger: Started writing to {self.filename}")
        
    def log_emg(self, connection_id, emg_data):
        """
        Queue EMG data for writing
        
        Args:
            connection_id: Connection ID
            emg_data: List/array of EMG values (8 channels)
        """
        timestamp = time.time()
        device_name = self.device_names.get(connection_id, f"unknown_{connection_id}")
        
        try:
            # Non-blocking put with timeout
            self.data_queue.put({
                'timestamp': timestamp,
                'device_name': device_name,
                'connection_id': connection_id,
                'data_type': 'EMG',
                'data_values': emg_data
            }, timeout=0.001)
        except queue.Full:
            print(f"WARNING: CSV queue full, dropping EMG sample")
            
    def log_imu(self, connection_id, imu_data):
        """
        Queue IMU data for writing
        
        Args:
            connection_id: Connection ID
            imu_data: Dict with 'orientation', 'accel', 'gyro' keys
        """
        timestamp = time.time()
        device_name = self.device_names.get(connection_id, f"unknown_{connection_id}")
        
        try:
            self.data_queue.put({
                'timestamp': timestamp,
                'device_name': device_name,
                'connection_id': connection_id,
                'data_type': 'IMU',
                'data_values': imu_data
            }, timeout=0.001)
        except queue.Full:
            print(f"WARNING: CSV queue full, dropping IMU sample")
            
    def _writer_loop(self):
        """
        Main writer loop running in separate thread
        """
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                data = self.data_queue.get(timeout=0.1)
                
                # Format data values as string
                if isinstance(data['data_values'], dict):
                    # IMU data
                    data_str = str(data['data_values'])
                else:
                    # EMG data - convert to comma-separated string
                    data_str = ','.join(map(str, data['data_values']))
                
                # Write row
                self.csv_writer.writerow([
                    data['timestamp'],
                    data['device_name'],
                    data['connection_id'],
                    data['data_type'],
                    data_str
                ])
                
                # Flush every 100 samples for reasonable performance
                if self.data_queue.qsize() < 100:
                    self.csv_file.flush()
                    
            except queue.Empty:
                # No data, continue waiting
                self.csv_file.flush()  # Flush on idle
                continue
                
    def stop(self):
        """Stop the CSV writer and close file"""
        print("CSV Logger: Stopping...")
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.writer_thread:
            self.writer_thread.join(timeout=2.0)
            
        # Write remaining data in queue
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                data_str = str(data['data_values'])
                self.csv_writer.writerow([
                    data['timestamp'],
                    data['device_name'],
                    data['connection_id'],
                    data['data_type'],
                    data_str
                ])
            except queue.Empty:
                break
                
        # Close file
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            
        print(f"CSV Logger: Closed {self.filename}")
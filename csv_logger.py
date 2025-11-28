"""
Synchronized CSV Logger with timer-based sampling
Features:
- Single CSV file with all Myo data in one row
- Timer-based sampling at user-specified rate (default 200 Hz)
- Latest-value buffering with NaN for missing data
- Non-blocking updates to minimize data collection impact
- Background thread for file writing
"""
import csv
import threading
import time
from datetime import datetime
import math


class CSVLogger:
    """
    Timer-based CSV logger that synchronizes data from multiple Myos.
    Samples at fixed intervals and writes all sensor data in one row.
    
    REASONING: User wants all data (EMG + IMU from all Myos) in single row,
    sampled at consistent intervals (200 Hz default) regardless of actual
    sensor arrival times.
    """
    
    def __init__(self, filename_prefix="myo_data", sampling_rate=200, num_myos=2):
        """
        Initialize synchronized CSV logger
        
        Args:
            filename_prefix: Base name for CSV file
            sampling_rate: Sampling frequency in Hz (default: 200)
            num_myos: Number of Myo armbands to expect (default: 2)
            
        REASONING: From config.py, default MYO_AMOUNT = 2. Sampling rate
        configurable via command line for flexibility.
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{filename_prefix}_{timestamp}.csv"
        
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1.0 / sampling_rate  # seconds between samples
        self.num_myos = num_myos
        
        # Threading control
        self.stop_event = threading.Event()
        self.timer_thread = None
        self.write_thread = None
        
        # Device tracking: connection_id -> device_name
        # REASONING: User wants device_name (not MAC), stored when Myo connects
        self.device_names = {}  # {0: "Myo_ABC123", 1: "Myo_DEF456"}
        
        # Latest sensor values for each Myo
        # REASONING: Timer thread reads from here, sensor handlers update here
        # Using NaN as initial value per user requirement
        self.latest_data = {}
        for i in range(num_myos):
            self.latest_data[i] = {
                'name': 'Unknown',
                'emg_sample1': [float('nan')] * 8,  # ch0-ch7
                'emg_sample2': [float('nan')] * 8,  # ch0-ch7
                'roll': float('nan'),
                'pitch': float('nan'),
                'yaw': float('nan'),
                'accel_x': float('nan'),
                'accel_y': float('nan'),
                'accel_z': float('nan'),
                'gyro_x': float('nan'),
                'gyro_y': float('nan'),
                'gyro_z': float('nan')
            }
        
        # Lock for thread-safe access to latest_data
        self.data_lock = threading.Lock()
        
        # Write buffer for batch operations
        self.write_buffer = []
        self.buffer_lock = threading.Lock()
        self.batch_size = 100  # Write every 100 samples
        
        # CSV file handle
        self.csv_file = None
        self.csv_writer = None
        
        # Performance monitoring
        self.total_samples = 0
        self.start_time = None
        self.missed_deadlines = 0
        
    def register_device(self, connection_id, device_name):
        """
        Register a device name for a connection ID
        
        Args:
            connection_id: Connection ID (0, 1, 2...)
            device_name: Device name from Myo
            
        REASONING: Called from myodriver.py after device info is read.
        Stores device_name which appears in every CSV row.
        """
        with self.data_lock:
            if connection_id < self.num_myos:
                self.device_names[connection_id] = device_name
                self.latest_data[connection_id]['name'] = device_name
                print(f"CSV Logger: Registered device '{device_name}' as Myo {connection_id}")
            else:
                print(f"WARNING: Connection ID {connection_id} exceeds expected number of Myos ({self.num_myos})")
        
    def start(self):
        """Start the CSV logger with timer and writer threads"""
        self.start_time = time.time()
        
        # Initialize CSV file with dynamic header based on num_myos
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Build header
        # REASONING: Header format requested by user - all data in one row
        # Each Myo gets: name + 16 EMG (2 samples × 8 channels) + 9 IMU
        header = ['timestamp']
        for i in range(self.num_myos):
            header.append(f'myo{i}_name')
            # EMG Sample 1 (channels 0-7)
            for ch in range(8):
                header.append(f'myo{i}_emg_s1_ch{ch}')
            # EMG Sample 2 (channels 0-7)
            for ch in range(8):
                header.append(f'myo{i}_emg_s2_ch{ch}')
            # IMU data
            header.extend([
                f'myo{i}_roll', f'myo{i}_pitch', f'myo{i}_yaw',
                f'myo{i}_accel_x', f'myo{i}_accel_y', f'myo{i}_accel_z',
                f'myo{i}_gyro_x', f'myo{i}_gyro_y', f'myo{i}_gyro_z'
            ])
        
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        
        # Start timer thread (samples at fixed rate)
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()
        
        # Start writer thread (handles buffered file writes)
        self.write_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.write_thread.start()
        
        print(f"CSV Logger: Started")
        print(f"  File: {self.filename}")
        print(f"  Sampling rate: {self.sampling_rate} Hz ({self.sampling_interval*1000:.2f} ms interval)")
        print(f"  Number of Myos: {self.num_myos}")
        print(f"  Batch size: {self.batch_size} samples")
        
    def update_emg(self, connection_id, emg_sample1, emg_sample2):
        """
        Update latest EMG data (non-blocking)
        
        Args:
            connection_id: Connection ID
            emg_sample1: Tuple/list of 8 EMG values (first sample)
            emg_sample2: Tuple/list of 8 EMG values (second sample)
            
        REASONING: Called from data_handler.py:handle_emg() with both samples.
        Non-blocking update to avoid impacting real-time data collection.
        """
        if connection_id >= self.num_myos:
            return
            
        with self.data_lock:
            self.latest_data[connection_id]['emg_sample1'] = list(emg_sample1)
            self.latest_data[connection_id]['emg_sample2'] = list(emg_sample2)
            
    def update_imu(self, connection_id, imu_data):
        """
        Update latest IMU data (non-blocking)
        
        Args:
            connection_id: Connection ID
            imu_data: Dict with keys: roll, pitch, yaw, accel_x/y/z, gyro_x/y/z
            
        REASONING: Called from data_handler.py:handle_imu() with parsed IMU data.
        """
        if connection_id >= self.num_myos:
            return
            
        with self.data_lock:
            self.latest_data[connection_id].update({
                'roll': imu_data['roll'],
                'pitch': imu_data['pitch'],
                'yaw': imu_data['yaw'],
                'accel_x': imu_data['accel_x'],
                'accel_y': imu_data['accel_y'],
                'accel_z': imu_data['accel_z'],
                'gyro_x': imu_data['gyro_x'],
                'gyro_y': imu_data['gyro_y'],
                'gyro_z': imu_data['gyro_z']
            })
            
    def _timer_loop(self):
        """
        Timer loop that samples data at fixed intervals
        
        REASONING: User requirement 1C + 2 - independent timer samples all data
        at specified rate, using latest available values.
        """
        next_sample_time = time.time()
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # Check if we're behind schedule
            if current_time > next_sample_time + self.sampling_interval:
                self.missed_deadlines += 1
                if self.missed_deadlines % 100 == 0:
                    print(f"WARNING: Missed {self.missed_deadlines} sampling deadlines")
            
            # Wait until next sample time
            sleep_time = next_sample_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Sample all data at this moment
            timestamp = time.time()
            row = self._capture_row(timestamp)
            
            # Add to write buffer
            with self.buffer_lock:
                self.write_buffer.append(row)
            
            self.total_samples += 1
            
            # Schedule next sample
            next_sample_time += self.sampling_interval
            
    def _capture_row(self, timestamp):
        """
        Capture current state of all Myos as a single row
        
        REASONING: Reads latest_data atomically and builds row with all Myo data.
        Uses current values or NaN if no data received yet.
        """
        row = [timestamp]
        
        with self.data_lock:
            for i in range(self.num_myos):
                data = self.latest_data[i]
                
                # Add device name
                row.append(data['name'])
                
                # Add EMG Sample 1 (ch0-ch7)
                row.extend(data['emg_sample1'])
                
                # Add EMG Sample 2 (ch0-ch7)
                row.extend(data['emg_sample2'])
                
                # Add IMU data
                row.extend([
                    data['roll'], data['pitch'], data['yaw'],
                    data['accel_x'], data['accel_y'], data['accel_z'],
                    data['gyro_x'], data['gyro_y'], data['gyro_z']
                ])
        
        return row
    
    def _writer_loop(self):
        """
        Writer loop that handles buffered file writes
        
        REASONING: Separate thread for file I/O to avoid blocking timer thread.
        Batch writes reduce disk I/O overhead.
        """
        while not self.stop_event.is_set():
            time.sleep(0.1)  # Check buffer every 100ms
            
            with self.buffer_lock:
                if len(self.write_buffer) >= self.batch_size:
                    self._flush_buffer()
                    
    def _flush_buffer(self):
        """
        Flush write buffer to disk
        
        REASONING: Batch writing reduces file I/O overhead. Called when buffer
        reaches batch_size or during shutdown.
        """
        if self.write_buffer:
            self.csv_writer.writerows(self.write_buffer)
            self.csv_file.flush()
            self.write_buffer.clear()
            
    def get_stats(self):
        """Get performance statistics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'elapsed_seconds': elapsed,
            'total_samples': self.total_samples,
            'missed_deadlines': self.missed_deadlines,
            'actual_rate': self.total_samples / elapsed if elapsed > 0 else 0,
            'target_rate': self.sampling_rate,
            'buffer_size': len(self.write_buffer)
        }
    
    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        print("\n=== CSV Logger Statistics ===")
        print(f"Elapsed time: {stats['elapsed_seconds']:.1f}s")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Target rate: {stats['target_rate']} Hz")
        print(f"Actual rate: {stats['actual_rate']:.1f} Hz")
        print(f"Missed deadlines: {stats['missed_deadlines']}")
        print(f"Buffer size: {stats['buffer_size']}")
        
    def stop(self):
        """Stop the CSV logger and close file"""
        print("\nCSV Logger: Stopping...")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.timer_thread:
            self.timer_thread.join(timeout=2.0)
        if self.write_thread:
            self.write_thread.join(timeout=2.0)
            
        # Flush remaining buffer
        print("CSV Logger: Flushing remaining data...")
        with self.buffer_lock:
            self._flush_buffer()
        
        # Print final statistics
        self.print_stats()
        
        # Close file
        if self.csv_file:
            self.csv_file.close()
            
        print(f"CSV Logger: Closed {self.filename}")
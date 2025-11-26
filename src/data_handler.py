import csv
import time
from datetime import datetime
import struct

class DataHandler:
    """
    Unified 2-arm Myo data handler.
    Writes ONE CSV row only when both arms have new EMG + IMU data.
    """

    def __init__(self, config_obj):
        self.config = config_obj
        self.start_time = time.time()

        # Store last data for each Myo
        self.last_emg = {0: None, 1: None}
        self.last_imu = {0: None, 1: None}

        # Map Myo BLE connections to IDs
        self.connection_to_myo_id = {}

        # Unified CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"myo_2arm_unified_{timestamp}.csv"
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # CSV header
        header = ['timestamp']
        for col in [1, 2]:
            header += [
                f'col{col}_emg1', 'col{}_emg2'.format(col), 'col{}_emg3'.format(col), 'col{}_emg4'.format(col),
                'col{}_emg5'.format(col), 'col{}_emg6'.format(col), 'col{}_emg7'.format(col), 'col{}_emg8'.format(col),
                f'col{col}_quat_w', f'col{col}_quat_x', f'col{col}_quat_y', f'col{col}_quat_z',
                f'col{col}_acc_x', f'col{col}_acc_y', f'col{col}_acc_z',
                f'col{col}_gyro_x', f'col{col}_gyro_y', f'col{col}_gyro_z'
            ]

        self.csv_writer.writerow(header)
        print(f"\nCreated 2-arm unified CSV: {filename}\n")

    def register_connection(self, conn, myo_id):
        self.connection_to_myo_id[conn] = myo_id
        print(f"Registered connection {conn} as Myo {myo_id}")

    def get_myo_id(self, payload):
        conn = payload.get('connection', None)
        if conn is None:
            return 0

        if conn not in self.connection_to_myo_id:
            new_id = len(self.connection_to_myo_id)
            if new_id >= self.config.MYO_AMOUNT:
                new_id = 0
            self.register_connection(conn, new_id)

        return self.connection_to_myo_id[conn]

    # ---------------- EMG ----------------
    def handle_emg(self, payload):
        myo_id = self.get_myo_id(payload)

        if 'emg' in payload:
            emg = payload['emg']
        elif 'value' in payload:
            emg = struct.unpack('<8b', payload['value'][:8])
        else:
            return

        self.last_emg[myo_id] = list(emg)
        self.try_write_row()

    # ---------------- IMU ----------------
    def handle_imu(self, payload):
        myo_id = self.get_myo_id(payload)

        if 'value' in payload:
            raw = payload['value']
            orientation = [x / 16384.0 for x in struct.unpack('<4h', raw[0:8])]
            accel = [x / 2048.0 for x in struct.unpack('<3h', raw[8:14])]
            gyro = [x / 16.0 for x in struct.unpack('<3h', raw[14:20])]
        else:
            return

        self.last_imu[myo_id] = orientation + accel + gyro
        self.try_write_row()

    # -------------- WRITE ROW WHEN READY --------------
    def try_write_row(self):
        """
        Write CSV only when BOTH arms have:
        - EMG
        - IMU
        """

        # Check if all data exists
        if None in (self.last_emg[0], self.last_imu[0], self.last_emg[1], self.last_imu[1]):
            return

        # Prepare row
        timestamp = time.time() - self.start_time

        row = (
            [timestamp] +
            self.last_emg[0] + self.last_imu[0] +
            self.last_emg[1] + self.last_imu[1]
        )

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # Clear stored data so next row uses fresh packets
        self.last_emg = {0: None, 1: None}
        self.last_imu = {0: None, 1: None}

    def close_files(self):
        print("\nClosing unified CSV...")
        try:
            self.csv_file.close()
        except:
            pass

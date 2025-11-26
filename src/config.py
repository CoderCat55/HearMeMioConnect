"""
Default values for the script. Can be overridden by system args.
"""
from src.public.myohw import *


class Config:

    MYO_AMOUNT = 2  # Default amount of myos to expect
    EMG_MODE = 0x02  # 0x02 = send raw EMG data (200Hz)
    IMU_MODE = 0x01  # 0x01 = send IMU data (accel, gyro, orientation)
    CLASSIFIER_MODE = 0x01  # 0x00 = disabled

    DEEP_SLEEP_AT_KEYBOARD_INTERRUPT = False  # Turn off connected devices after keyboard interrupt

    PRINT_EMG = True  # Console print EMG data
    PRINT_IMU = True # Console print IMU data

    VERBOSE = False  # Verbose console
    GET_MYO_INFO = True  # Get and display myo info at sync

    MESSAGE_DELAY = 0.1  # Added delay before every message sent to the myo

    OSC_ADDRESS = "127.0.0.1"  # localhost
    OSC_PORT = 57120  # default port

    RETRY_CONNECTION_AFTER = 2  # Reconnection timeout in seconds
    MAX_RETRIES = None  # Max amount of retries after unexpected disconnect

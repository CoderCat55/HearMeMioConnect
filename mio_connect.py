from src.myodriver import MyoDriver
from src.config import Config
from csv_logger import CSVLogger
import serial
import getopt
import sys


def main(argv):
    config = Config()
    csv_filename = "myo_data"
    enable_csv = False
    sampling_rate = 200  # Default 200 Hz

    # Get options and arguments
    try:
        opts, args = getopt.getopt(argv, 'hsn:a:p:v', [
            'help', 'shutdown', 'nmyo=', 'address=', 'port=', 'verbose', 
            'csv', 'csv-prefix=', 'sampling-rate='
        ])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
        
    turnoff = False
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_usage()
            sys.exit()
        elif opt in ('-s', '--shutdown'):
            turnoff = True
        elif opt in ("-n", "--nmyo"):
            config.MYO_AMOUNT = int(arg)
        elif opt in ("-a", "--address"):
            config.OSC_ADDRESS = arg
        elif opt in ("-p", "--port"):
            config.OSC_PORT = int(arg)
        elif opt in ("-v", "--verbose"):
            config.VERBOSE = True
        elif opt == '--csv':
            enable_csv = True
        elif opt == '--csv-prefix':
            csv_filename = arg
            enable_csv = True
        elif opt == '--sampling-rate':
            sampling_rate = int(arg)
            enable_csv = True

    # Run
    myo_driver = None
    csv_logger = None
    
    try:
        # Initialize CSV logger if enabled
        # REASONING: Create and start CSV logger BEFORE connecting Myos so it's
        # ready to receive device registrations and data updates
        if enable_csv:
            csv_logger = CSVLogger(
                filename_prefix=csv_filename,
                sampling_rate=sampling_rate,
                num_myos=config.MYO_AMOUNT
            )
            csv_logger.start()
        
        # Initialize Myo driver with CSV logger
        myo_driver = MyoDriver(config, csv_logger=csv_logger)

        # Connect to Myos
        myo_driver.run()

        if turnoff:
            # Turn off
            myo_driver.deep_sleep_all()
            return

        if Config.GET_MYO_INFO:
            # Get info and register with CSV logger
            # REASONING: This calls bluetooth.read_device_name() which populates
            # myo.device_name, then registers each Myo with csv_logger
            myo_driver.get_info()

        print("Ready for data.")
        print()

        # Receive and handle data
        # REASONING: This is the main loop that calls bluetooth.receive() which
        # triggers handle_emg() and handle_imu() callbacks
        while True:
            myo_driver.receive()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    except serial.serialutil.SerialException:
        print("ERROR: Couldn't open port. Please close MyoConnect and any program using this serial port.")

    finally:
        print("Disconnecting...")
        
        # Stop CSV logger first to flush all data
        if csv_logger:
            csv_logger.stop()
            
        # Disconnect Myos
        if myo_driver is not None:
            if Config.DEEP_SLEEP_AT_KEYBOARD_INTERRUPT:
                myo_driver.deep_sleep_all()
            else:
                myo_driver.disconnect_all()
                
        print("Disconnected")


def print_usage():
    """
    Print usage information
    
    REASONING: Added new CSV-related options while maintaining original options
    """
    message = """usage: python mio_connect.py [options]

Options and arguments:
    -h | --help                     Display this message
    -s | --shutdown                 Turn off (deep_sleep) the expected amount of myos
    -n | --nmyo <amount>            Set the amount of devices to expect (default: 2)
    -a | --address <address>        Set OSC address (default: localhost)
    -p | --port <port_number>       Set OSC port (default: 3000)
    -v | --verbose                  Get verbose output
    --csv                           Enable CSV logging (default: disabled)
    --csv-prefix <filename>         Set CSV filename prefix (default: myo_data)
    --sampling-rate <hz>            Set CSV sampling rate in Hz (default: 200)

Examples:
    # Run without CSV logging:
    python mio_connect.py -n 2
    
    # Run with CSV logging at 200 Hz (default):
    python mio_connect.py -n 2 --csv
    
    # Run with CSV logging at 100 Hz:
    python mio_connect.py -n 2 --csv --sampling-rate 100
    
    # Run with custom CSV filename:
    python mio_connect.py -n 2 --csv-prefix experiment1 --sampling-rate 200
    
    # Turn off connected devices:
    python mio_connect.py -n 2 -s
"""
    print(message)
    
if __name__ == "__main__":
    main(sys.argv[1:])
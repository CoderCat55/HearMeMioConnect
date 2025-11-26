from src.myodriver import MyoDriver
from src.config import Config
from csv_logger import CSVLogger
import serial
import getopt
import sys


def main(argv):
    config = Config()
    csv_filename = "myo_data"

    # Get options and arguments
    try:
        opts, args = getopt.getopt(argv, 'hsn:a:p:v', ['help', 'shutdown', 'nmyo', 'address', 'port', 'verbose', 'csv='])
    except getopt.GetoptError:
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
            config.OSC_PORT = arg
        elif opt in ("-v", "--verbose"):
            config.VERBOSE = True

    # Run
    myo_driver = None
    csv_logger = None
    try:
        csv_logger = CSVLogger(filename_prefix=csv_filename)
        csv_logger.start()
        
        # Init
        myo_driver = MyoDriver(config, csv_logger=csv_logger)  # CHANGE: Pass CSV logger

        # Connect
        myo_driver.run()

        if turnoff:
            # Turn off
            myo_driver.deep_sleep_all()
            return

        if Config.GET_MYO_INFO:
            # Get info
            myo_driver.get_info()
        
        for myo in myo_driver.myos:
            if myo.device_name:
                csv_logger.register_device(myo.connection_id, myo.device_name)


        print("Ready for data.")
        print()

        # Receive and handle data
        while True:
            myo_driver.receive()

    except KeyboardInterrupt:
        print("Interrupted.")

    except serial.serialutil.SerialException:
        print("ERROR: Couldn't open port. Please close MyoConnect and any program using this serial port.")

    finally:
        print("Disconnecting...")
        if csv_logger:
            csv_logger.stop()
        if myo_driver is not None:
            if Config.DEEP_SLEEP_AT_KEYBOARD_INTERRUPT:
                myo_driver.deep_sleep_all()
            else:
                myo_driver.disconnect_all()
        print("Disconnected")


def print_usage():
    # CHANGE: Updated usage message with CSV option
    message = """usage: python mio_connect.py [-h | --help] [-s | --shutdown] [-n | --nmyo <amount>] [-a | --address \
<address>] [-p | --port <port_number>] [-v | --verbose] [-c | --csv <filename>]

Options and arguments:
    -h | --help: display this message
    -s | --shutdown: turn off (deep_sleep) the expected amount of myos
    -n | --nmyo <amount>: set the amount of devices to expect
    -a | --address <address>: set OSC address
    -p | --port <port_number>: set OSC port
    -v | --verbose: get verbose output
    -c | --csv <filename>: set CSV filename prefix (default: myo_data)
"""
    print(message)
    
if __name__ == "__main__":
    main(sys.argv[1:])

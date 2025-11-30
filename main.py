from mioconnect.src.myodriver import MyoDriver
from mioconnect.src.config import Config
import serial
import getopt
import sys


def main(argv):
    config = Config()

    # Get options and arguments
    try:
        opts, args = getopt.getopt(argv, 'hsn:a:p:v', ['help', 'shutdown', 'nmyo', 'address', 'port', 'verbose'])
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
    try:
        # Init
        myo_driver = MyoDriver(config)

        # Connect
        myo_driver.run()

        if turnoff:
            # Turn off
            myo_driver.deep_sleep_all()
            return

        if Config.GET_MYO_INFO:
            # Get info
            myo_driver.get_info()

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
        if myo_driver is not None:
            if Config.DEEP_SLEEP_AT_KEYBOARD_INTERRUPT:
                myo_driver.deep_sleep_all()
            else:
                myo_driver.disconnect_all()
        print("Disconnected")


def Classify():
    print("meow")
    #tell sckit to classify and return the data

def Calibrate(gesturename):
    print("meow")
    #should we handle this here or anywhere else?


def Train():
    print("meow")
    #tell sckit to train the itself

def Command():
    value = input("Enter your command majesty: ")
    match value:
        case "train":
            print("now will run train function")
            Train()
        case "classify":
            print("now will run classify function")
            Classify()
        case "calibrate":
            print("now will run calibrate function")
            Calibrate()
        case _:
            print("doğru düzgün komut yazsana biz böyle mi çalışacaz?")

if __name__ == "__main__":
    main(sys.argv[1:])

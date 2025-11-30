"""
This is the main file of my code.
Functions: create a mdns webserver named "hearme" as a hotspot. in raspi
import important things from mio connect 

"""
from mioconnect.src.myodriver import MyoDriver
from mioconnect.src.config import Config
#import webserberlibrary
"""I used to create a AP in esp32 but raspberyy probobaly can connect to phone's internet and maintain http protocol?
Is this true if so how?

because otherwise it wants additional dongle ? why

I need my raspberyy pi to open a http server which we can conenct even though we are in a place which doesnt have any wifi networks
https://raspberrypi.stackexchange.com/questions/117526/how-to-create-a-wireless-and-own-rpi-wifi-network

"""

"""
might add getdata endpoint for showing data inside the app (not now)
"""
"""

"""
#init 
config = Config()
myo_driver = MyoDriver(config)

#http post or get requests refernce from old code 
def webserver():
    #initialize mdns server with "hearme"
    print("meow")
    endpoints= [ 
         'classify',
         'stopclassify'
        'calibrate',
         'disconnect',
         'connect',
         'train',
         'getdata'
                ]
    """These endpoint trigger relative functions """

    
def Classify():
    print("meow")

def Stopclassify():
     print("meow")

def Calibrate(gestureidORname):
    print("meow")

def Connect():
    # Connect
        myo_driver.run()

def Disconnect():
    myo_driver.disconnect_all()
    print("meow")

def Train():
    print("meow")

def Command():
    print("meow")

def Getdata():
    #this triggers to get data 
    print("meow")
 
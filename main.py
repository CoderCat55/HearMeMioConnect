"""handles multiprocessing
handles user commands
"""
def Classify():
    print("meow")
    #tell sckit to classify and return the data
    #should I open a class in sckit and call classify method here?

def Calibrate(gestureName):
    print("meow")
    #saves calibration samples for 3 seconds
    #should we handle this here or anywhere else?


def Train():
    print("meow")
    #tell sckit to train the itself
    #should I open a class in sckit and call train method here?


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
            gesture_name = input("Welche gesture möchten Sie calibrate?")
            Calibrate(gesture_name)
        case _:
            print("doğru düzgün komut yazsana biz böyle mi çalışacaz?")

if __name__ == "__main__":
    print("lets gooo")
    #initialize getdata will run  
    #then
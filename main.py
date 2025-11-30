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
    # 1. Set flag in shared memory to record for 3 seconds
    # 2. After 3 seconds, data is in shared memory as numpy array
    # 3. Pass to classifier
    classifier.add_calibration_sample(gesture_name, recorded_data)
    # 4. Save to disk
    np.save(f'calibration_data/{gesture_name}.npy', recorded_data)


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
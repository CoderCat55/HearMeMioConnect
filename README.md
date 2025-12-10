Readmeeee
Okay as you can see we have a c++ code and a python code
c++ code is just for reference it used to be my old code. Now we are improving it
We got raspberyy pi 5 instead of esp32 and myo armbands (2) instead of other sensors.
We will be running more advanced ML like svm instead of knn.
also we will be recognizing dynamic gestures not just statics so we need some kind of time seriss of data
This is a real time system. 

this part is already done [main.py the file user will run also where user will give commands frequently
it initializes processes. getdata process and main.py is always running
sckit-learn would do classificaiton,training etc. when triggered
I need to have a class for sckitlearn which has methods like classify,train then call them inside main .py
Also I need to save data (calibration) for gestures for 3 seconds. Since I will be doing dynamic and static gestures we need a time series or some time related things in order to use.
Also we need to get data not using connection id we need to get data with device name)]


Solution 4: Match Window Sizes
Option A: Use same duration for both (RECOMMENDED)
python
# main.py
CALIBRATION_DURATION = 3.0  # seconds
CLASSIFICATION_DURATION = 3.0  # same!
Issue 5: Window Size Mismatch

Calibration: 3 seconds of data (~300 samples at 100Hz)

Classification: 1 second of data (~100 samples)

Model extracts: Statistical features (mean, std, min, max, range)

Problem: Statistical features from 3-second windows will have completely different distributions than 1-second windows!

Example:

3-second gesture: Mean EMG = 50, Std = 20

Same gesture, 1-second window: Mean EMG = 70, Std = 35

Model trained on 3s data cannot recognize 1s patterns!


Q2: "Do I need to perform the gesture before classifying it?"
Answer: With your current code: NO, but you SHOULD!
Current behavior:
Classify function has a 5-second countdown
After countdown, it grabs the last 3 seconds from the buffer
It classifies whatever happened in those 3 seconds
Problem:
If you sit still → classifies "no gesture" data
If you moved randomly → classifies random movements
If you performed gesture → classifies the gesture ✅
Papers recommend: Use motion detection to segment gestures automatically!

///////for this 
delay 5sn 
classify function çağrıldı . classify fucntionu 3sn beklicek sonra shared memoryden alıcak 

Issue 3: Timestamp Column Mismatch 🔴 CRITICAL
Chain of thought:
Looking at data_handler.py line 156:
python
self.stream_buffer[idx] = sample  # Writes 34 features
But looking at data_handler.py line 161:
python
combined_row = np.concatenate(([timestamp], sample))  # Creates 35-column row
self.calib_buffer[self.calib_index.value] = combined_row
So:
Stream buffer: Has 34 features (NO timestamp column)
Calibration buffer: Has 35 features (WITH timestamp in column 0)
But looking at main.py line 27:
python
shm_stream = shared_memory.SharedMemory(name=stream_mem_name)
stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 35), dtype=np.float32, ...)
Stream buffer has 35 columns allocated, but only 34 are written!
And in get_recent_data_from_shared_mem (line 89), you return the full buffer including column 0 which is NEVER WRITTEN (always 0).
Then in model.py line 25 extract_features:
python
for channel in range(time_series_data.shape[1]):  # Iterates over ALL columns
If you pass stream buffer data (35 cols with col 0 = 0), you extract features from 35 channels including the zero column = 175 features. If you pass calibration buffer data (35 cols with col 0 = timestamp), you extract features from 35 channels including timestamp = 175 features.
But your model was trained on calibration data with timestamp in column 0, so:
Training features: Extract from columns [timestamp, 34 sensors] = 175 features (timestamp gets mean/std/min/max/range)
Classification features: Extract from columns [0, 34 sensors] = 175 features (zeros get mean/std/min/max/range)
RESULT: Feature distribution mismatch! Model sees different feature patterns between training and testing.
SOLUTION 1: Write timestamp to stream buffer too:
python
# data_handler.py - line 156
for timestamp, sample in self.local_buffer:
    idx = self.stream_index.value % len(self.stream_buffer)
    # Write timestamp + sample (35 columns total)
    combined_row = np.concatenate(([timestamp], sample))
    self.stream_buffer[idx] = combined_row
    self.stream_index.value += 1
    
    # Also write to calibration buffer if recording
    if self.recording_flag.value == 1:
        if self.calib_index.value < len(self.calib_buffer):
            self.calib_buffer[self.calib_index.value] = combined_row
            self.calib_index.value += 1
SOLUTION 2 (BETTER): Don't include timestamp in features at all:
python
# model.py - line 25
@staticmethod
def extract_features(time_series_data):
    """
    Extract statistical features from time series
    time_series_data shape: (time_steps, 35) where col 0 is timestamp
    Returns: 1D feature vector
    """
    # Extract ONLY sensor data (columns 1-34), ignore timestamp
    sensor_data = time_series_data[:, 1:]  # Shape: (time_steps, 34)
    
    features = []
    for channel in range(sensor_data.shape[1]):  # Now 34 channels
        channel_data = sensor_data[:, channel]
        
        # Statistical features
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.min(channel_data),
            np.max(channel_data),
            np.max(channel_data) - np.min(channel_data),
        ])
    
    # Total features: 34 channels * 5 features = 170 features
    return np.array(features)
I recommend Solution 2 because timestamp shouldn't be used as a feature anyway!
//so ı just need timestamps for debugging right model wont be using them?
//aslında Türklerin çalışması bize çok yol gösterici olucak çünkü onlarda gesture based olarak yapmışlar sınıflandırmayı zamana bağlı değil de.

what I want to do:
1- add a timestamp column inside numpy array instead of as a filename. Because we need to see which data we get based on timestamps

Please really read all the code. 
Do not make assumptions while answering.
While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me. 

Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.




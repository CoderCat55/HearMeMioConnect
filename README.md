Readmeeee
Okay as you can see we have a c++ code and a python code
c++ code is just for reference it used to be my old code. Now we are improving it
We got raspberyy pi 5 instead of esp32 and myo armbands (2) instead of other sensors.
We will be running more advanced ML like svm instead of knn.
also we will be recognizing dynamic gestures not just statics so we need some kind of time seriss of data
This is a real time system. 

main.py the file user will run also where user will give commands frequently
it initializes processes. getdata process and main.py is always running
sckit-learn would do classificaiton,training etc. when triggered
I need to have a class for sckitlearn which has methods like classify,train then call them inside main .py
Also I need to save data (calibration) for gestures . Since I will be doing dynamic and static gestures we need a time series or some time related things in order to use.
Also we need to get data not using connection id we need to get data with device name
calibrate
    save data as a numpy array for sckitlearn 
    get the gesturename  from user 
    triggers realtime data getting using mioconnect for a specified time then saves data to shared memory
----------------- already done left to gain basic understanding of the system --------------------------

My aim: 
"Segement the data in calibration_data/p3 according to rest gesture in the calibration_data/p3rest folder. Because  gestures in calibration_data/p3 include rest->gesture->rest So we need to cut the rest positions and then save these new data to calibration_data/p3new." Note: one-class SVM might work for this

SO I need this procedure for all participant their folders are named as follows: p1,p2,p3,p4,p5,p6 and p1rest,p2rest.....
This procedure will be done between same participant with this I mean segment p3 data with p3rest, not p3 data with p4rest
I need detailed plan on how would you implement this.

for classification details :
while is_running True
            check if the gesture is rest  with restmodel(model 1) which is a svm model  windowSize 20ms  
            if gesture != rest
                do feature engineering  
                run classifymodel(model 2) window size 100 ms 

we will have 2 models which they will need seperate classes restmodel and classifymodel class:
model1(the rest model) will be one-class SVM calibration_data/pXrest (X being participant number 1 from 6 (all the participants )) as rest class.

model2(the classification model) will be trained on calibration_data/pXnew (X being participant number 1 from 6 (all the participants ) each gesture name as a class.


classification and commanding should run simulteneusly so I can start-stop classsification. I would prefer multiprocessing.

Problems I've Identified:

Current model.py has only ONE classifier class - You need TWO separate classes (RestModel and ClassifyModel)
Feature extraction happens at wrong granularity - You want 20ms windows for rest detection but current code works on full 3-second samples
No sliding window implementation - Need to process data in overlapping/sequential windows
Segmentation is offline preprocessing - Should be separate from real-time classification

Open Questions for You

Segmentation threshold: When sliding through gesture data, how many consecutive "non-rest" windows should trigger a segment start? (e.g., 3 consecutive non-rest = gesture begins)
Window overlap: Should 20ms windows overlap (sliding) or be sequential (tumbling)?
Minimum segment length: What's the minimum length for a valid gesture segment? (e.g., discard segments < 0.5 seconds)
Training strategy: Should rest model be participant-specific or trained on all participants' rest data combined?
Feature engineering: Your current features (Line 23-35 model.py) are mean, std, min, max, range. Do you want to add frequency domain features (FFT) for better gesture discrimination?


Notes: each model can have its own train funciton

RULES: Please really read all the code. Do not make assumptions while answering. While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me.

Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.

When change of code only change what is relevant do not try to change anything that is not relevant with my aim. I need to know which parts you have made changes tell me because sometimes you change something inside a fucntion which I am not aware of.



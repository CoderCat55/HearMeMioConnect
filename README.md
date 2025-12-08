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


    calibrate

        save data as a numpy array for sckitlearn 
        get the gesturename  from user 
        triggers realtime data getting using mioconnect for a specified time then saves data to shared memory

what I want to do:
1- save numpy array for 3 seconds the data order should never change. Like the first values should be from myoITU etc. I mean use device names.
2- currently we are saving a numpy arrray add that numpy array a coulmn at its begining which will store the gesture label which retrieved from user.
3- These numpy arrays will be saved into dataframe inside model train function.

Which changes should be made in which part of the code.
We need to go step by step.

Please really read all the code. 
Do not make assumptions while answering.
While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me. 

Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.




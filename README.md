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
So I guess I need to have a class for sckitlearn which has methods like classify,train then call them inside main .py? Is this a reasonable and efficent way or do you have any other suggestions?
Also I need to save data (calibration) for gestures for 3 seconds. Since I will be doing dynamic and static gestures we need a time series or some time related things in order to use.
Also we need to get data not using connection id we need to get data with device name


    calibrate
        save data as a numpy array for sckitlearn 
        get the gesturename  from user 
            triggers realtime data getting using mioconnect for a specified time then saves data to shared memory

shared buffer and index is not defined in myo driver. also please control my code can I expect it to run as I aimed shortly and detailly list all the other changes needed

Please really read all the code. 
Do not make assumptions while answering.
While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me. 
Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.




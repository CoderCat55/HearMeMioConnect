Readmeeee
Okay as you can see we have a c++ code and a python code
c++ code is just for reference it used to be my old code. Now we are improving it
We got raspberyy pi 5 instead of esp32 and myo armbands (2) instead of other sensors.
We will be running more advanced ML like svm instead of knn.
also we will be recognizing dynamic gestures not just statics so we need some kind of time seriss of data
This is a real time system. 

Should I use multiprocessing for getdata,sckit and main?
because I want to communicate with user all times and I dont want to block other jobs with this.
I will trigger sckitlearn related things with webserver like give it command to do classification and returning the result also training itself with new data.

main.py the file user will run also where user will give commands frequently
it initializes processes. getdata process and main.py is always running
sckit-learn would do classificaiton,training etc. when triggered
So I guess I need to have a class for sckitlearn which has methods like classify,train then call them inside main .py? Is this a reasonable and efficent way or do you have any other suggestions?
Also I need to save data (calibration) for gestures for 3 seconds. Since I will be doing dynamic and static gestures we need a time series or some time related things in order to use it I guess we need to modify data_handler.py for to do this?
Also we need to get data not using connection id we need to get data with device name


    calibrate
        save data as a numpy array for sckitlearn 
        get the gesturename  from user 
        Option A:
            triggers realtime data getting using mioconnect for a specified time then saves data to shared memory
        Option B:
            data is getting already in another process just go and save the data as a numpy array
        Option C:
            Should calibrate interval accessed to save only one data (like my old implementation)

Please really read all the code. 
Do not make assumptions while answering.
While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me. 
Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.

Which type of data sharing should be used among programs I think numpy arrays since they can be used by sckitlearn and fast
How should we store sensor values
What happens when datas come at different times? Would we give each a timestamp then what?
How should we store calibration samples?
    as a numpy array?
    in shared memory or inside sckitlearn
which of the mioconnect files would be used and does it requaire any modification



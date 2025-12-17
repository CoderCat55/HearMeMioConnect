Readmeeee
****** This part is for reference only 
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
Also I need to save data (calibration) for gestures for 3 seconds. Since I will be doing dynamic and static gestures we need a time series or some time related things in order to use.
Also we need to get data not using connection id we need to get data with device name
*************************

My aim:
to write a webserver.py which can integrate to other parts of this code in harmony 

The problems:
I need to send json datas with /data endpoint with current datas from shared memory. Note: Do not effect shared memory


RULES:
Please really read all the code. 
Do not make assumptions while answering.
While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me. 

Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.

When change of code only change what is relevant do not try to change anything that is not relevant with my aim.
I need to know which parts you have made changes tell me because sometimes you change something inside a fucntion which I am not aware of.




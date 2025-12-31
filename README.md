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

classification and commanding should run simulteneusly so I can start-stop classsification. I would prefer multiprocessing.

classification details :
while is_running True
            check if the gesture is rest  with restmodel(model 1) window size 20 
            if gesture != rest
                do feature engineering  
                run classifymodel(model 2) window size 100 ms 

we will have 2 models which they will need seperate classes restmodel and classifymodel class:
model1(the rest model) in rest_model1.py It can accurately recognize if a gesture is rest or not.

model2(the classification model) will be trained on processed_data ,all the participants each gesture name as a class.

----------------- already done left to gain basic understanding of the system --------------------------

TODO:
Currentlty model2 classifies a fixed amount of windowsize. What I want to do is capture the gesture datas betweeen two rest positions then pass it to model2 for feature enginering and classificaiton. So when position is not rest (the gesture is started) the datas will be copied into classification buffer and when position is rest again indicating the gesture has finished. The datas in the classification buffer will be given to gesture_model an result will be returned

Lets critizse the framework at the bottom
while is_runnning:
    rest_model.predict()
    if is_rest:
        gstarted=False
        continue
    if !is_rest:
        gstarted=True
        while gstarted:
            #save current data to classifcaiton buffer
        #pass datas to gesture model


DO NOT FORGET THE RULES
Create a list how this would be implemented to current system , which parts should be changed which parts should be added and where. 


RULES: Please really read all the code. Do not make assumptions while answering. While giving ansswers include the chain of thought, why did you make that assumption, which part of thee code leads you to that? If you are not sure about how something works just tell me.

Your aim is to discuss the structure with me. I need you to be objective. You should use strategies like listing pros and cons of a situation and judge it according to my aim.

When change of code only change what is relevant do not try to change anything that is not relevant with my aim. I need to know which parts you have made changes tell me because sometimes you change something inside a fucntion which I am not aware of.

You may only write which parts of the code I should change and where changes shhould be made to save time instead of writing the whole script again.

WHAT IS EXPECTED FROM THIS CODE. WHAT CAN IT DO. WHICH MISMATCHES HAVE YOU SPOTTED AS A LIST


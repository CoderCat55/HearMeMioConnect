Readmeeee
Okay as you can see we have a c++ code and a python code
c++ code is just for reference it used to be my old code. Now we are improving it
We got raspberyy pi 5 instead of esp32 and myo armbands (2) instead of other sensors.
We will be running more advanced ML like svm instead of knn.
also we will be recognizing dynamic gestures not just statics so we need some kind of time seriss of data

Should I use multiprocessing for getdata,sckit and main?
because I want to communicate with web server all times and I dont want to block other jobs with this.
I will trigger sckitlearn related things with webserver like give it command to do classification and returning the result also training itself with new data.

main.py
import everything from mioconnect
functions
    webserver
        open mdns server with "hearme"
        create webserver    
        classify endpoint
        calibrate """
        connect """
        disconnect ""
        train  ""
    classify
        trigger sckitlearn to classify
        2 options:
            continue classification until stopclassify is called or 
            access the classify endpoint triggerin this function at a fixed interval (like my old code)

    calibrate
        save data as a numpy array for sckitlearn 
        get the gesture id/name 
        Option A:
            triggers realtime data getting using mioconnect for a specified time then saves data to shared memory
        Option B:
            data is getting already in another process just go and save the data as a numpy array

        Option C:
            Should calibrate interval accessed to save only one data (like my old implementation)

    connect
    disconnect
    train
    command

Here is how to get device name. I dont want to use conenction ID because armbands can connect at different orders.

  def _get_device_name(self, connection_id):
        """Get device name from connection ID"""
        for myo in self.myo_driver.myos:
            if myo.connection_id == connection_id:
                return myo.device_name
        return None

# Get device name from MyoDriver
device_name = self._get_device_name(connection_id)
    if device_name is None:
        return  # Myo info not available yet



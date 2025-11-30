"""
trains itself based on calibration samples
reads realtime data from shared memory for a fixed time and classifies returns result
"""
from sklearn import svm
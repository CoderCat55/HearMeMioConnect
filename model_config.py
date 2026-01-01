"""Configuration settings for HearMeMioConnect"""

# Buffer Sizes
STREAM_BUFFER_SIZE = 250   # ~5 seconds at 50Hz
CALIBRATION_BUFFER_SIZE = 150  # 3 seconds at 50Hz

# Duration Settings (seconds)
CALIBRATION_DURATION = 3  
CLASSIFICATION_DURATION = 3 

# Countdown Settings
CALIBRATION_STARTS = 5 
CLASSIFICATION_STARTS = 5 

# Gesture Recognition Settings
GESTURE_TIMEOUT_SAMPLES = 150  # 3 seconds max gesture duration at 50Hz
MIN_GESTURE_SAMPLES = 5        # Minimum samples for valid gesture

# Model Parameters
REST_WINDOW_SIZE = 20
WINDOW_SIZE_MS = 300
SAMPLING_RATE = 50
CONFIDENCE_THRESHOLD = 0.0
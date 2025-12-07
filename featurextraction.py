import numpy as np

def extract_features_from_window(window):
    """
    Extracts a set of time-domain features from a window of sensor data using
    vectorized operations for efficiency.

    Args:
        window (np.array): A 2D numpy array of shape (window_size, num_channels).
                           For this project, this is (200, 34).

    Returns:
        np.array: A 1D numpy array of shape (num_features,).
                  num_features = num_channels * 5 = 170.
    """
    # Ensure window is a numpy array
    window = np.asarray(window)

    # 1. Mean Absolute Value (MAV)
    mav = np.mean(np.abs(window), axis=0)

    # 2. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(window**2, axis=0))

    # 3. Variance (VAR)
    var = np.var(window, axis=0)

    # 4. Zero Crossings (ZC) - Corrected implementation
    # Counts crossings of the zero line, not the window mean.
    zc = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0)

    # 5. Slope Sign Changes (SSC)
    # Calculate difference along the time axis (axis=0)
    dx = np.diff(window, axis=0)
    ssc = np.sum(np.diff(np.sign(dx), axis=0) != 0, axis=0)

    # Concatenate all features for all channels into a single feature vector
    # The order is [mav_ch1, mav_ch2, ..., rms_ch1, rms_ch2, ...]
    features = np.concatenate((mav, rms, var, zc, ssc))

    return features
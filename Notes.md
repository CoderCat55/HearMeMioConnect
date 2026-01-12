Processed data yerine yogurt_duzelt_n 'deki rows_delted klasörünü ekledim

example numpy array path : rows_deleted\p1\bek_1765449146.npy
1) gesture model should train on rows_deleted/p{1-6}/all npy arrays that dont start with "rest"
2) Rest model should be like this:
- rest class: rows_deleted/p{1-6}/all npy arrays that start with "rest"
- non-rest class: rows_deleted/p{1-6}/all npy arrays that dont start with "rest"

belki classify functionunu process olarak değil thread olarak yapmak daha iyi olaiblirdiyada belki bir config dosyası ekleyip parametreleri oradan çağırabiliriz

not training ile classificationw windowları aynı boyutta olmalı 

Classification logic: When non-rest is detected, the system now waits 1.5 seconds, extracts 75 samples from the circular buffer, applies sliding windows of 20 samples with 50% overlap to generate 6 predictions, and uses majority voting to determine the final gesture with confidence percentage.


[sohbet gemini](https://gemini.google.com/share/3cd4bd572f33)
If you downsample the EMG to 50 Hz to match the IMU, you are throwing away 75% of your muscle data. This will make your gesture recognition significantly less accurate because muscle "spikes" happen very fast—much faster than 50 times a second.

Here is the breakdown of your two options:

Option 1: Upsample IMU (Recommended)
You keep the EMG at 200 Hz. For every 4 EMG samples, you only have 1 IMU sample. You simply "fill in the blanks" for the IMU.

How: Use Linear Interpolation (drawing a straight line between two IMU points) or Zero-order Hold (simply repeating the last known IMU value until a new one arrives).

Pros: You keep the high-resolution muscle data. This is critical for detecting quick finger snaps or rapid hand movements.

Cons: Your IMU data looks "smoother" than it really is, but since arm rotation is slow compared to muscle firing, this doesn't hurt your model.

Option 2: Downsample EMG (Not Recommended)
You throw away 3 out of every 4 EMG samples to match the 50 Hz IMU rate.

How: Decimation (keeping every 4th sample).

Pros: Less data to process, smaller file sizes.

Cons: Aliasing. High-frequency muscle activity will be "folded" into low frequencies, creating noise. Your model will struggle to tell the difference between a "fist" and a "rest" because the characteristic "shimmer" of a tensed muscle is lost at 50 Hz.

Averaging the EMG samples to match the IMU frequency (downsampling from 200 Hz to 50 Hz) is a common technique, but in the specific case of muscle data (EMG), it has a major drawback: Averaging acts like a "Blur" filter.

Why Averaging EMG is Risky
EMG signals are "bipolar," meaning the raw values fluctuate rapidly between positive and negative. If you take a simple mathematical average of raw EMG data, the positive and negative peaks will cancel each other out, leaving you with a value close to zero. You would effectively erase the signal you're trying to measure.


If you really want to downsample to 50 Hz to save processing power or match the IMU exactly, you should use the Root Mean Square (RMS) or Mean Absolute Value (MAV) instead of a simple average.
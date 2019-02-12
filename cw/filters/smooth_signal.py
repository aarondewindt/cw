import numpy as np
from scipy.signal import butter, filtfilt


def smooth_signal(signal, wn=0.01):
    """
    Smooths a signal using a 5th order Butterworth low-pass filter.

    :param signal: Signal to be filtered.
    :param wn: Cuttof frequency
    :return: Filtered signal
    """
    butter_params = butter(5, [wn])
    mean_signal = np.mean(signal)
    clip_signal = np.hstack((mean_signal, signal, mean_signal))
    return filtfilt(*butter_params, clip_signal)[1:-1]

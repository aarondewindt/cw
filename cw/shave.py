import numpy as np
from scipy.signal import butter, filtfilt


__author__ = "Aaron M. de Windt"


# TODO: Write unittest for shave function.


def lerp(x, x0, x1, y0, y1):
    """
    Interpolates to get the value of y for x by linearly interpolating
    between points (x0, y0) and (x1, y)

    :return:
    """
    t = (x-x0)/(x1 - x0)
    return (1 - t) * y0 + t * y1


def next_good(bad_samples_idx, i):
    """
    Find the index of the next good item in the list.

    :param bad_samples_idx: List of the indices of the bad samples.
    :param i: Index of the current item.

    :return: Index of the next good item.
    """

    while True:
        i += 1
        if i >= (len(bad_samples_idx) - 1):
            return bad_samples_idx[-1] + 1

        if (bad_samples_idx[i] + 1) == bad_samples_idx[i + 1]:
            continue
        else:
            break

    return bad_samples_idx[i] + 1


def prev_good(bad_samples_idx, i):
    """
    Find the index of the previous good item in the list.

    :param bad_samples_idx: List of the indices of the bad samples.
    :param i: Index of the current item.

    :return: Index of the previous good item.
    """

    while True:
        i -= 1
        if i < 1:
            return bad_samples_idx[0] - 1

        if (bad_samples_idx[i] - 1) == bad_samples_idx[i - 1]:
            continue
        else:
            break

    return bad_samples_idx[i] - 1


def shave(signal, factor=4, wn=0.005, plot=False, clip=None):
    """
    This function reduces the number of peaks in a signal. It does this by first
    passing the signal through a Butterworth low-pass filter to get a trend. Then
    the a standard deviation of the signal is calculated and a band is by
    adding/substracting the trend with a standard deviation times a factor.

    :param signal: The signal to be shaved
    :param factor: The factor used to multiply the standard deviation when calculating
       the allowed band
    :param wn: Cutoff frequency of the Butterworth low-pass filter.
    :param plot: True to plot the original signal, new signal and band
    :param clip: List with the low and high clip values.

    :return: Shaved signal.
    """

    if clip is not None:
        clip_signal = np.clip(signal, *clip)
        bad_samples = (signal > clip[1]) | (signal < clip[0])
    else:
        clip_signal = signal
        bad_samples = [False]*len(clip_signal)

    butter_params = butter(5, wn)
    mean_signal = np.mean(clip_signal)
    clip_signal = np.hstack((mean_signal, clip_signal, mean_signal))
    trend = filtfilt(*butter_params, clip_signal)[1:-1]
    clip_signal = clip_signal[1:-1]
    core_signal = clip_signal - trend
    std_signal = np.std(core_signal)

    upper_bound = trend + factor * std_signal
    lower_bound = trend - factor * std_signal

    bad_samples |= (signal > upper_bound) | (signal < lower_bound)

    bad_samples_idx = np.where(bad_samples)[0]

    for i, i_signal in enumerate(bad_samples_idx):
        i_0 = prev_good(bad_samples_idx, i)
        i_1 = next_good(bad_samples_idx, i)

        if i_0 >= len(bad_samples):
            i_0 = i_1
            i_1 = next_good(bad_samples_idx, bad_samples_idx.index(i_1 - 1))

        if i_1 < 0:
            i_1 = i_0
            i_0 = prev_good(bad_samples_idx, bad_samples_idx.index(i_0))

        clip_signal[i_signal] = lerp(i_signal, i_0, i_1, clip_signal[i_0], clip_signal[i_1])

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(signal, "b")
        plt.plot(clip_signal, "y")
        plt.plot(upper_bound, "g")
        plt.plot(lower_bound, "g")
        plt.plot(trend, "m")
        plt.plot(bad_samples_idx, signal[bad_samples_idx], ".r")
        plt.show()

    return clip_signal

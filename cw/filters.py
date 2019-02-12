import numpy as np
from scipy.signal import butter, filtfilt


# TODO: Write unittests for filters.


class ExponentialRunningAverage:
    """
    Class used to calculate the exponential running average of a data stream.
    The type is not limited to floats. This class supports any type that
    supports basic arithmetic (subtraction, addition and multiplication).

    :param initial: Optional initial value.
    :param weight: Weight to give to new values being added.
    """

    def __init__(self, initial=None, weight=0.8):
        self.value = initial  #: The current value of the running average.
        self.weight = weight  #: The weight of the running average.

    def update(self, value):
        """
        Update the running average with a new entry.

        :param value: Value to be added.
        :return: The new value of the running average.
        """
        # If the value has not already been set, set it.
        if self.value is None:
            self.value = value
        else:
            # Calculate the new value.
            self.value = ((1-self.weight) * self.value + self.weight * value)
        return self.value

    def reset(self):
        """
        Reset the value to None.
        """
        self.value = None


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

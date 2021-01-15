import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from cw.filters.exponential_running_average import ExponentialRunningAverage


def smooth_signal(signal, method="butter", **kwargs):
    """
    Smooths a signal using the given method.

    :param signal: Signal to be filtered.
    :param method: Method to be used. By default this is the
                   5th order butterworth low-pass filter.
    :param **kwargs: Extra parameters for the underlying method's function.
    :return: Filtered signal
    """
    if method in smoothing_methods:
        if isinstance(signal, xr.DataArray):
            data_array = signal
            signal = data_array.values
        else:
            data_array = None

        signal = smoothing_methods[method](signal, **kwargs)

        if data_array is not None:
            return xr.DataArray(
                data=signal,
                coords=data_array.coords,
                dims=data_array.dims,
                name=f"{data_array.name}_smoothed",
                attrs=data_array.attrs)
        else:
            return signal


def smooth_signal_butter(signal, wn=0.01):
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


def smooth_signal_exponential(signal, weight=0.8):
    """
    Smooths a signal using a exponential running average.

    :param signal: Signal to be filtered.
    :param weight: Weight
    :return: Filtered signal
    """
    signal = np.asarray(signal)
    era = ExponentialRunningAverage(weight=weight)
    new_signal = np.empty(signal.shape)
    for i in range(signal.shape[0]):
        new_signal[i, ...] = era.update(signal[i, ...])
    return new_signal


smoothing_methods = {
    "butter": smooth_signal_butter,
    "exp": smooth_signal_exponential
}

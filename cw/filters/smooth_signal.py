import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
from cw.filters.exponential_running_average import ExponentialRunningAverage


def smooth_signal(signal, method="butter", **kwargs):
    """
    Smooths a signal using the given method.

    :param signal: Signal to be filtered.
    :param method: Method to be used. Options {`butter`, `iir`, `exp`}. By default `butter`.
    :param kwargs: Extra parameters for the underlying method's function.
    :return: Filtered signal
    """
    if method in smoothing_methods:
        if isinstance(signal, xr.DataArray):
            data_array = signal
            signal = data_array.values

        elif isinstance(signal, xr.Dataset):
            smoothed_dataset = xr.Dataset()
            for key in signal:
                smoothed_data_array = smooth_signal(signal[key], method, **kwargs)
                smoothed_dataset[smoothed_data_array.name] = smoothed_data_array
            return smoothed_dataset

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


def smooth_signal_butter(signal, wn=0.01, order=5):
    """
    Smooths a signal using a Butterworth low-pass filter.
    Applies a forwards and backwards pass, thus the smoothed
    signal has zero-phase and twice the filter order.

    :param signal: Signal to be filtered.
    :param wn: Cuttof frequency.
    :param order: (half) Filter order.
    :return: Filtered signal.
    """
    butter_params = butter(order, [wn])
    mean_signal = np.mean(signal)
    clip_signal = np.hstack((mean_signal, signal, mean_signal))
    return filtfilt(*butter_params, clip_signal)[1:-1]


def smooth_iir(signal, b, a):
    """
    Smooths a signal using a IIR filter.
    Applies a forwards and backwards pass, thus the smoothed
    signal has zero-phase and twice the filter order.

    :param signal: Signal to be filtered.
    :param b: IIR filter b vectors.
    :param a: IIR filter a vectors
    :return: Filtered signal
    """
    mean_signal = np.mean(signal)
    clip_signal = np.hstack((mean_signal, signal, mean_signal))
    return filtfilt(b, a, clip_signal)[1:-1]


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
    "exp": smooth_signal_exponential,
    "iir": smooth_iir
}

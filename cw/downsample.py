import numpy as np
from scipy.interpolate import interp1d


def downsample(time, signal, new_time):
    """
    Downsamples a signal.

    :param time: Signal time vector
    :param signal: Signal values
    :param new_time: New time vector or the new time vector step size.
    :return:
    """
    # time = time if isinstance(time, np.ndarray) else np.array(time)
    # data = data if isinstance(data, np.ndarray) else np.array(data)

    data_interp = interp1d(time, signal, axis=0, copy=False)
    if np.isscalar(new_time):
        new_time = np.arange(time[0], time[-1], new_time)

    new_data = data_interp(new_time)

    return new_time, new_data

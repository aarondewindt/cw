import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import Optional


class Plotter:
    def __init__(self):
        pass

    def process_results(self, result: xr.Dataset):
        for state_name in result:
            self.default_plotter(result[state_name])

        plt.show()

    def default_plotter(self, data: xr.DataArray, sup_title: Optional[str]=None, figsize=(15, 7)):
        fig = plt.figure(figsize=figsize)
        data_values = data.values
        data_t = data.t.values
        plt.title(data.name)
        if sup_title:
            plt.suptitle(sup_title)

        if np.ndim(data_values) != 1:
            # Shape the the values, the first dimension is time.
            shape = np.shape(data_values)[1:]
            legend = []
            for idx in product(*(range(dim_size) for dim_size in shape)):
                plt.plot(data_t, data_values[(slice(None), *idx)])
                legend.append(", ".join(map(str, idx)))
            plt.legend(legend)
        return fig

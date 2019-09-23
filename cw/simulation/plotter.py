import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product
from typing import Optional
from pathlib import Path


class Plotter:
    def __init__(self):
        pass

    def plot(self, result: xr.Dataset, sup_title=None, figsize=(15, 7)):
        for state_name in result:
            self.default_plotter(result[state_name], sup_title, figsize)

    def plot_to_pdf(self, path: Path, result: xr.Dataset, sup_title=None, figsize=(15, 7)):
        with PdfPages(str(path)) as pdf_file:
            for state_name in result:
                fig = self.default_plotter(result[state_name], sup_title, figsize)
                pdf_file.savefig(fig)
                plt.close(fig)

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
        else:
            plt.plot(data_t, data_values)
        return fig

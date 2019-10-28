import xarray as xr
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product
from typing import Optional
from pathlib import Path


class Plotter:
    def __init__(self):
        pass

    def plot(self, result: xr.Dataset, sup_title=None, figsize=(15, 7), states=None):
        if states is None:
            states = result.data_vars

        for state_name in states:
            if state_name in result:
                self.default_plotter(result[state_name], sup_title, figsize)

    def plot_to_pdf(self, path: Path, result: xr.Dataset, sup_title=None, figsize=(15, 7), states=None):
        if states is None:
            states = result.data_vars
        with PdfPages(str(path)) as pdf_file:
            for state_name in states:
                if state_name in result:
                    figs = self.default_plotter(result[state_name], sup_title, figsize)
                    for fig in figs:
                        pdf_file.savefig(fig)
                        plt.close(fig)

        print(f"Results plot saved at: {path.absolute()}")

    def default_plotter(self, data: xr.DataArray, sup_title: Optional[str]=None, figsize=(15, 7)):
        fig = plt.figure(figsize=figsize)
        data_values = data.values
        data_t = data.t.values
        plt.title(data.name)
        if sup_title:
            plt.suptitle(sup_title)

        line_styles = ["-", "--", ":", "-."]

        if np.ndim(data_values) != 1:
            # Shape the the values, the first dimension is time.
            shape = np.shape(data_values)[1:]
            legend = []
            for idx in product(*(range(dim_size) for dim_size in shape)):
                plt.plot(data_t, data_values[(slice(None), *idx)], line_styles[idx[0] % 4])
                legend.append(", ".join(map(str, idx)))
            plt.legend(legend)
            return fig,
        else:
            if data.dtype == np.quaternion:
                plt.plot(data_t, quaternion.as_float_array(data_values))
                plt.legend(["q0", "q1", "q2", "q3"])
                fig2 = plt.figure(figsize=figsize)
                plt.title(f"{data.name} (euler)")
                plt.plot(data_t, quaternion.as_euler_angles(data_values))
                plt.legend(["x", "y", "z"])
                return fig, fig2
            else:
                plt.plot(data_t, data_values)
                return fig,

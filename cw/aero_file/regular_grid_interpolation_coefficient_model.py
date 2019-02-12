from cw.aero_file.coefficient_model_base import CoefficientModelBase
from cw.cached import cached
from collections import OrderedDict
import scipy.interpolate as interpolate
import numpy as np
from typing import Dict


class RegularGridInterpolationCoefficientModel(CoefficientModelBase):
    """
    Aerodynamic coefficient model based on scipy's :class:`scipy.interpolate.RegularGridInterpolator`.

    :param numpy.ndarray table: ND-table holding the aerodynamic coefficients.
    :param string method: The method of interpolation to perform. Supported are "linear" and
        "nearest".
    :param bool bounds_error: If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
    :param number fill_value: If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.
    """

    model_name = "regular_grid_interpolation"

    def __init__(self, table, parameters, *, method="linear", bounds_error=False, fill_value=np.nan, **kwargs):
        self.table = np.array(table)
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.parameters: Dict[str, list] = parameters if isinstance(parameters, OrderedDict) else OrderedDict(parameters)

        self.__parameter_names = list(self.parameters.keys())
        self.interpolator = interpolate.RegularGridInterpolator(
                [param_values for _, param_values in self.parameters.items()], table, method, bounds_error, fill_value)

    @cached
    def parameter_names(self):
        """
        List of strings with the names of the parameters for each dimension (alpha, mach, etc,).
        """
        return self.__parameter_names

    def get_coefficient(self, *args, **kwargs):
        """
        Returns the coefficients corresponding to the conditions defined by the
        parameters passed. These parameters depend on the table stored and parameters defined.

        .. note::

           You must either pass all parameters named or unnamed. Mixing named and unnamed
           parameters is not supported.

        """
        if args:
            # Call the interpolator and pass the values in args.
            if len(args) != len(self.parameter_names):
                args = args[:len(self.parameter_names)]
            return self.interpolator(np.array(args))[0]
        else:
            # 1. Get a list with the values for each parameter in the same order as the
            #    one defined in self.parameter_names. This line will also raise an
            #    exception if kwargs does not contain all parameters defined in
            #    self.parameter_names
            # 2. Call the interpolator and return the resulting value.
            return self.interpolator(np.array([kwargs[param_name] for param_name in self.parameter_names]))[0]

    def dump_data(self):
        """
        Returns a dictionary containing the parameters to be saved
        to the data file. These are the same parameters that the constructor takes in as input.
        """
        return {
            "table": self.table.tolist(),
            "method": self.method,
            "bounds_error": self.bounds_error,
            "fill_value": self.fill_value,
            "parameters": [list(x) for x in self.parameters.items()]
        }

    def point_value_tables(self, points=None):
        """
        Returns two tables with the point coordinates (parameter values), and coefficient values.
        Each row in the table corresponds to a specific data point. It can be used as the input
        for the CoKringing aerodynamics code.

        :param list points: Optional list of points in the point value tables.
        :return: Tuple with two :class:`numpy.ndarray` instances with the point coordinates and
           coefficient value tables respectively.
        """
        # If the points are given, run the default implementation of this function. This
        # will call get_coefficient for each point and return the tables.
        if points is not None:
            return super().point_value_tables(points)

        # Initialize tables
        param_table = np.empty((self.table.size, len(self.parameters)))
        data_table = np.empty((self.table.size,))

        # Loop through all elements in the coefficient table.
        for i, (param_idxs, coef_value) in enumerate(np.ndenumerate(self.table)):
            # Get the value of each parameter for this specific element in the coefficient table.
            param_table[i, :] = [param_values[param_idx] for param_idx, (param_name, param_values) in
                                 zip(param_idxs, self.parameters.items())]

            # Put the coefficient in the data table.
            data_table[i] = coef_value

        return param_table, data_table

    @property
    def default_alpha(self):
        """
        Default angle of attack to use if the aerodynamic model doesn't have
        an angle of attack input.
        """
        return 0.0

    @property
    def default_beta(self):
        """
        Default side-slip angle to use if the aerodynamic model doesn't have a
        side-slip angle input.
        """
        return 0.0

    @property
    def default_mach(self):
        """
        Default mach number to use if the aerodynamic model doesn't have a
        mach number input. This is the smallest mach number available in the
        tables.
        """
        if "mach" in self.parameters:
            return min(self.parameters['mach'])


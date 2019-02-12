from cw.aero_file.coefficient_model_base import CoefficientModelBase
from cw.cached import cached
import scipy.interpolate as interpolate
import numpy as np


class LinearNDInterpolationCoefficientModel(CoefficientModelBase):
    """
    Aerodynamic coefficient model based on scipy's
    :class:`scipy.interpolate.LinearNDInterpolator`.

    :param list parameter_names: List of strings with the names of the parameters
       corresponding to each dimension.
    :param numpy.ndarray points: Data point coordinates, or a precomputed scipy
       Delaunay triangulation.
    :param numpy.ndarray values: Data values.
    :param number fill_value: Value used to fill in for requested points outside
       of the convex hull of the input points. If not provided, then the default is nan.
    :param bool rescale: Rescale points to unit cube before performing interpolation.
       This is useful if some of the input dimensions have incommensurable units and
       differ by many orders of magnitude.
    """

    model_name = "linear_nd_interpolation"

    def __init__(self, parameter_names, points, values, fill_value=float('nan'), rescale=False):
        self.__parameter_names = list(parameter_names)
        self.points = np.array(points)
        self.values = np.array(values)
        self.fill_value = fill_value
        self.rescale = rescale

        self.interpolator = interpolate.LinearNDInterpolator(
            points,
            values,
            fill_value,
            rescale
        )

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
            return self.interpolator(np.array(args))
        else:
            # 1. Get a list with the values for each parameter in the same order as the
            #    one defined in self.parameter_names. This line will also raise an
            #    exception if kwargs does not contain all parameters defined in
            #    self.parameter_names
            # 2. Call the interpolator and return the resulting value.
            # TODO: fix this
            raise NotImplementedError()
            return self.interpolator(tuple([kwargs[param_name] for param_name in self.parameter_names]))

    def dump_data(self):
        """
        Returns a dictionary containing the parameters to be saved
        to the data file. These are the same parameters that the constructor takes in as input.
        """
        return {
            "parameter_names": list(self.parameter_names),
            "points": self.points.tolist(),
            "values": self.values.tolist(),
            "fill_value": self.fill_value,
            "rescale": self.rescale
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

        return self.points, self.values

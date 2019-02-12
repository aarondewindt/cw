from abc import ABCMeta, abstractmethod
from typing import Union, Dict


class CoefficientModelBase(metaclass=ABCMeta):
    """
    Base class used for aerodynamic models.
    """

    @property
    @abstractmethod
    def model_name(self):
        """
        String with the aerodynamic method name. This property should be static.
        """
        pass

    @property
    @abstractmethod
    def parameter_names(self) -> set:
        """
        List of strings with the names of the parameters for each dimension (alpha, mach, etc,).
        """
        pass

    @abstractmethod
    def get_coefficient(self, **kwargs):
        """
        Returns the coefficients corresponding to the conditions defined by the
        parameters passed. These parameters depend per method, coefficient, and
        data source.

        .. note::

           You must either pass all parameters named or unnamed. Mixing named and unnamed
           parameters is not supported.

        """
        pass

    @abstractmethod
    def dump_data(self):
        """
        Returns a dictionary containing the parameters to be saved
        to the data file. The constructor should take these parameters as
        input.
        """
        pass

    @property
    def default_alpha(self) -> Union[float, None]:
        """
        Default angle of attack to use if the aerodynamic model doesn't have an
        angle of attack input. When overridden it must either return a float or None if
        no default angle of attack is possible.
        """
        return None

    @property
    def default_beta(self) -> Union[float, None]:
        """
        Default side-slip angle to use if the aerodynamic model doesn't have a
        side-slip angle input. When overridden it must either return a float or None if
        no default side-slip angle is possible.
        """
        return None

    @property
    def default_mach(self) -> Union[float, None]:
        """
        Default mach number to use if the aerodynamic model doesn't have a
        mach number input. When overridden it must either return a float or None if
        no default mach number is possible.
        """
        return None

    def point_value_tables(self, points=None):
        """
        Returns two tables with the point coordinates (parameter values), and coefficient values.
        Each row in the table corresponds to a specific data point. It can be used as the input
        for the Cokringing aerodynamics code.

        .. warning::

           The points parameter is NOT optional for some aerodynamic models. For example
           the CoKringing model.

        :param list points: Optional list of points in the point value tables. This parameter
           is not optional for some aerodynamics models.
        :return: Tuple with two :class:`numpy.ndarray` instances with the point coordinates and
           coefficient value tables respectively.

        """
        if points is None:
            raise ValueError(
                "Points parameter is not optional for the '{}' aerodynamic model".format(self.model_name))
        else:
            # Return the points and the coefficient for that point
            return points, [self.get_coefficient(*point) for point in points]

    @staticmethod
    def validate_data(properties):
        """
        When overridden, the function must validate the properties dictionary. If the dictionary is
        valid it must return None object. Otherwise it must return a string with an error message.

        :param properties: Properties dictionary to be validated.
        :return: None if valid, otherwise a string with an error message.
        """
        # The default behaviour to to mark everything as valid.
        return None

    @property
    def default_values(self) -> Dict[str, Union[float, None]]:
        """
        Default values to use when the aerodynamic model is missing an input parameter. It's not necessary
        to override this function since its values depend on the the other default_### function in this
        class.

        :return:
        """

        default_values = {
            "alpha": self.default_alpha,
            "beta": self.default_beta,
            "mach": self.default_mach
        }

        return default_values

from cw.aero_file.regular_grid_interpolation_coefficient_model import RegularGridInterpolationCoefficientModel
from cw.aero_file.coefficient_model_base import CoefficientModelBase
from cw.aero_file.linear_nd_interpolation_coefficient_model import LinearNDInterpolationCoefficientModel

aero_models_registry = {
    RegularGridInterpolationCoefficientModel.model_name: RegularGridInterpolationCoefficientModel,
    LinearNDInterpolationCoefficientModel.model_name: LinearNDInterpolationCoefficientModel
}

from cw.aero_file.aero_file import AeroFile
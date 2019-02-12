from cw.cached import cached_class
from cw.aero_file import aero_models_registry
from cw.aero_file.coefficient_model_base import CoefficientModelBase
from cw.aero_file import RegularGridInterpolationCoefficientModel
from cw.flex_file import flex_load, flex_dump
from cw.print_code import print_code
from cw.serializers import msgpack, yaml

import os
from datetime import datetime
from dateutil.parser import parse as datetime_parse
from collections import OrderedDict
from pathlib import Path
from io import StringIO

import pickle
import numpy as np
from cerberus import Validator
from typing import Union, Optional, Dict, Iterable


# These import are used by the generated code, so your IDE might complain about them being unused.
from cw.transformations import tr_ab
from cw.conversions import dcm_to_euler, rot_to_angle_2d, angle_to_rot_2d


valid_parameters = [
    "alpha",
    "mach",
    "beta"
]


class AeroFile:
    """
    This class is used to load in and create files holding aerodynamic tables.

    :param path: Path to the file to open. If None, the object will be opened
       in write only mode.
    :param mode:
    :param serializer: Default serializer. If None, MessagePack will be the default.
    """

    def __init__(self, path: Union[str, os.PathLike], *, mode: str="r"):
        # If no path was given then create this AeroFile object in write only mode.
        self.mode = mode

        # If the path is a string, create a PurePath object.
        self.path = path if isinstance(path, Path) else Path(path)
        """
        Path to the data file.
        """

        self.case = ''  #: Case name
        self.description = ''  #: Short description about the data
        self.created = datetime.today()
        """:class:`datetime.datetime` representing the time at which the data was created."""
        self.lref = np.nan  #: Reference length
        self.latref = np.nan  #: Lateral Reference length
        self.mrc = np.ones((3, 1)) * np.nan  #: Moment reference center.
        self.sref = np.nan  #: Reference area

        self.aero_parameter_names = None

        self.coefficients: Dict[str, CoefficientModelBase] = {}

        """
        Dictionary holding the coefficient objects for each coefficient.
        """

        self.output_dimensions: int = None
        self.code_obj = None

        self.__load_data()

    @cached_class
    def validator(self):
        """
        Cerberus validator object used to validate the aero_files being read.
        """
        # Load in schema, create a cerberus validator object and return it.
        schema = flex_load(abs_path("aero_file_schema.yaml"))
        return Validator(schema)

    def validate_data(self, data):
        """
        Validates the data coming in from the file by checking it against
        a cerberus schema.

        :param data: Data to be validated
        :return: A tuple with two elements. The first one is True if the
           data is valid, otherwise False. The second one contains a
           dictionary with the cerberus error information in case
           the file is not valid.
        """
        valid = self.validator.validate(data)
        return valid, self.validator.errors

    def __load_data(self):
        """
        Loads in the aerodynamic data from a file.

        :raises ValueError: If an unknown model was defined in the file.
        """

        if "r" in self.mode:
            # Load raw data from file.
            raw_data = flex_load(file_path=self.path,
                                 default_serializer=msgpack,
                                 default_is_gzipped=True)

            # Validate data, raise error if the data is invalid.
            valid, error = self.validate_data(raw_data)
            if not valid:
                raise ValueError("Invalid aerodynamic file.\n" + yaml.dump(error))

            # Read in parameters.
            self.case = raw_data['case']
            self.description = raw_data['description']
            self.lref = raw_data['lref']
            self.sref = raw_data['sref']
            self.latref = raw_data['latref']
            self.mrc = np.array(raw_data['mrc']).reshape((3, 1))

            # Load in the created datetime.
            # The datetime should be stored as a string in the data file, but
            # some serializer, like the yaml loader, will automatically parse it into a
            # datetime object.
            if issubclass(type(raw_data['created']), datetime):
                self.created = raw_data['created']
            else:
                self.created = datetime_parse(raw_data['created'])

            # Loop through the coefficient entries an create an instance of AeroMethodBase for each.
            for c_data in raw_data['coefficients']:
                # Get the class for the specified model.
                model_class = aero_models_registry.get(c_data['model'])

                # Check whether a class for the model was found. If not raise an exception.
                if model_class is None:
                    raise ValueError("Aerodynamic model '{}' unknown.".format(c_data['model']))
                else:
                    # Validate the properties dictionary.
                    result = model_class.validate_data(c_data['properties'])
                    if result is not None:
                        raise ValueError("Invalid aerodynamic file.\n" + result)

                    # If a class was found, create an instance of it and pass the parameters to the
                    # constructor and store the resulting object in the coefficients table.
                    coefficient_model = model_class(**c_data['properties'])

                    # Check whether are parameters are valid.
                    # TODO: Finish writing this.

                    self.add_coefficient(c_data['name'], coefficient_model)

    def add_coefficient(self, c_name, model):
        """
        Adds a coefficient model to the file.

        :param string c_name: The coefficient name.
        :param sim_common.file_formats.AeroModelBase model: The coefficient model.
        """
        # TODO: Add some checks here to check to for example check, coefficient name,  model type, etc.
        self.coefficients[c_name] = model

    def dump(self, path: Union[str, os.PathLike]=None):

        # The code after this function's definition uses this function to
        # dump the data to the file. The purpose of that code is to perform
        # checks and conversions before dumping.
        def dump_to_file(path: os.PathLike):
            # Create dictionary with the data to be stored to the file.
            aero_data = {
                "case": self.case,
                "description": self.description,
                "created": self.created.isoformat(),
                "lref": self.lref,
                "sref": self.sref,
                "latref": self.latref,
                "mrc": self.mrc.reshape((3,)).tolist(),
                "coefficients": [
                    {
                        "name": name,
                        "model": coef.model_name,
                        "properties": coef.dump_data()  # I refuse to call it data. -- Aaron M. de Windt
                    }
                    for name, coef in self.coefficients.items()]
            }

            # Validate the aero_data.
            valid, error = self.validate_data(aero_data)
            if not valid:
                raise ValueError("Error while dumping data.\n" + yaml.dump(error))

            # Open file and dump the data to it.
            flex_dump(file_path=path,
                      obj=aero_data,
                      default_serializer=msgpack,
                      default_is_gzipped=True)

        # Do checks and conversions.
        if path is not None:
            # Convert path to a path object if necessary.
            path = path if isinstance(path, Path) else Path(path)
            dump_to_file(path)
        else:
            # No path given, using self.path
            if "w" not in self.mode:
                raise PermissionError("AeroFile opened in read only mode.")
            else:
                dump_to_file(self.path)

    @classmethod
    def from_old_format(cls, path, no_deflections=True, case_name="Loaded from old file format file."):
        """
        Factory method that creates an instance of :class:`sim_common.file_formats.AeroFile`
        using a aerodynamic data file generated by the old tool.

        :param path: Path to the file to open.
        :param no_deflections: Remove the deflection dimensions from the data table.
        :return: Instance of :class:`sim_common.file_formats.AeroFile`
        """
        # Create a new AeroFile instance in write only mode.
        aero = cls(path, mode="w")

        # Read in the raw data using pickle.
        with open(aero.path, "rb") as f:
            raw_data = pickle.load(f)

        # Read in the parameters in the dictionary and put them in the new AeroFile instance
        aero.case = case_name
        aero.description = "Loaded in from a file using the old file format"
        aero.created = datetime.today()
        aero.lref = eval(raw_data['REFQ']['LREF'])[0]  # Prime example of Remon's lazininess. Store the number as a string in a list. Because many useful, no confusion.
        aero.sref = eval(raw_data['REFQ']['LATREF'])[0]
        aero.sref = eval(raw_data['REFQ']['SREF'])[0]
        aero.mrc = np.array([[eval(raw_data['REFQ']['XCG'])], [0], [0]])

        parameters = OrderedDict([(param_name, raw_data[param_name]) for param_name in
                                 ['alpha', 'beta', 'mach', 'deflctP', 'deflctQ', 'deflctR']])

        # The coefficient names in the old file format do not match the standard names. Standard
        # names can be found in the documentation.
        name_translation = {
            'cll': "c_ll",
            'cn': "c_z",
            "ca": "c_x",
            "cln": "c_ln",
            "cm": "c_m",
            "cy": "c_y"
        }

        # Old version of the old file format do not contain the data for c_ma. So we'll
        # only add it if present.
        if "cma" in raw_data:
            name_translation['cma'] = "c_m_a"

        # If the deflections are not being loaded in. Find the indices where the
        # deflections are 0 and remove them from the parameters list
        if no_deflections:
            p_idx = parameters['deflctP'].index(0)
            q_idx = parameters['deflctQ'].index(0)
            r_idx = parameters['deflctR'].index(0)
            parameters.pop('deflctP')
            parameters.pop('deflctQ')
            parameters.pop('deflctR')
        else:
            # If the deflections are being read in, then indices should be
            # limitless slices.
            p_idx = slice(None)
            q_idx = slice(None)
            r_idx = slice(None)

        # Loop through the coefficient name translation table.
        for c_name_old, c_name in name_translation.items():
            # Create a new RegularGridInterpolationAeroModel for each coefficient
            # and add it to the new aero_file.
            aero.add_coefficient(c_name, RegularGridInterpolationCoefficientModel(
                table=raw_data[c_name_old][:, :, :, p_idx, q_idx, r_idx],
                parameters=parameters
            ))

        return aero

    def calc_aerodynamic_forces_moments(self,
                                        *,
                                        rho: float,
                                        v_a: Union[np.ndarray, float],
                                        alpha: float=None,
                                        beta: float=None,
                                        dcm_ab: np.ndarray=None,
                                        mach: float=None,
                                        cg: Union[np.ndarray, float]=None,
                                        omega: Union[np.ndarray, float]=None):
        """

        :param rho:
        :param v_a:
        :param alpha:
        :param beta:
        :param mach:
        :param cg:
        :param omega:
        :return:
        """

        # If the code_obj has not been created yet, check what for parameters where
        # given to this functions and call 'setup_aerodynamic_forces_moments_calculation(...)'
        if self.code_obj is None:
            aerodynamic_model_input_parameters = set()

            if alpha is not None:
                aerodynamic_model_input_parameters.add("alpha")

            if beta is not None:
                aerodynamic_model_input_parameters.add("beta")

            if mach is not None:
                aerodynamic_model_input_parameters.add("mach")

            if dcm_ab is not None:
                aerodynamic_model_input_parameters.add("dcm_ab")

            output_dimensions = v_a.size

            self.setup_aerodynamic_forces_moments_calculation(
                input_parameters=aerodynamic_model_input_parameters,
                output_dimensions=output_dimensions
            )

        # If no coordinate was given for the center of gravity, use the
        # moment reference center.
        cg = self.mrc if cg is None else cg

        try:
            # Execute the generated code.
            exec(self.code_obj)  # Creates c_force and c_mom
        except:
            print_code(self.code_str)
            print("\n")
            raise

        # Calculate the aerodynamic forces and moments.
        # c_force and c_mom are declared in the exec above.
        v_a_abs = np.linalg.norm(v_a)
        f = 0.5 * rho * v_a_abs ** 2 * self.sref * c_force
        m = 0.5 * rho * v_a_abs ** 2 * self.sref * self.lref * c_mom

        return f, m

    def calc_forces_moments_coefficients(self,
                                         *,
                                         alpha: float=None,
                                         beta: float=None,
                                         dcm_ab: np.ndarray=None,
                                         mach: float=None,
                                         cg: Union[np.ndarray, float]=None,
                                         omega: Union[np.ndarray, float]=None,
                                         output_dimensions: int=None):
        """

        :param rho:
        :param v_a:
        :param alpha:
        :param beta:
        :param mach:
        :param cg:
        :param omega:
        :return:
        """

        # If the code_obj has not been created yet, check what for parameters where
        # given to this functions and call 'setup_aerodynamic_forces_moments_calculation(...)'
        if self.code_obj is None:
            aerodynamic_model_input_parameters = set()

            if alpha is not None:
                aerodynamic_model_input_parameters.add("alpha")

            if beta is not None:
                aerodynamic_model_input_parameters.add("beta")

            if mach is not None:
                aerodynamic_model_input_parameters.add("mach")

            if dcm_ab is not None:
                aerodynamic_model_input_parameters.add("dcm_ab")

            if output_dimensions is None:
                if cg is not None:
                    output_dimensions = cg.size
                elif omega is not None:
                    output_dimensions = omega.size
                elif dcm_ab is not None:
                    output_dimensions = len(dcm_ab)
                else:
                    raise AttributeError(
                        "Unable to determine the number of output dimensions. Please specify it.")

            self.setup_aerodynamic_forces_moments_calculation(
                input_parameters=aerodynamic_model_input_parameters,
                output_dimensions=output_dimensions
            )

        # If no coordinate was given for the center of gravity, use the
        # moment reference center.
        cg = self.mrc if cg is None else cg

        # Execute the generated code.
        results = locals()
        try:
            exec(self.code_obj, globals(), results)  # Creates c_force and c_mom
        except:
            print_code(self.code_str)
            print("\n")
            raise

        return results['c_force'], results['c_mom']

    def setup_aerodynamic_forces_moments_calculation(self,
                                                     input_parameters: Iterable,
                                                     output_dimensions: int,
                                                     default_alpha: Optional[float]=0.0,
                                                     default_beta: Optional[float]=0.0,
                                                     default_mach: Optional[float]=None):
        # Check if value of output_dimensions is valid
        if output_dimensions not in [1, 2, 3]:
            raise ValueError(f"'{output_dimensions}' is an invalid value for 'output_dimensions'.")

        self.output_dimensions = output_dimensions

        # Create string builder used to generate the code used to calculate the aerodynamic forces and moments.
        force_moment_calculation_code_builder = StringIO()

        # Put the default values in a dictionary, so they are more easily accessible later on.
        default_values = {
            'alpha': default_alpha,
            'beta': default_beta,
            'mach': default_mach
        }

        # Put the aerodynamic model input parameters into a set to make processing it easier.
        aero_parameter_names = set(input_parameters)

        # Dictionary holding the code used to calculate each coefficient.
        coefficients_code_dict: Dict[str, str] = {}

        in_aerodynamic_reference_frame = False

        # Check whether the dcm_ab or alpha/beta have to be calculated
        transformation_code: str = None

        # For the 2d case we need dcm_ab and alpha.
        if output_dimensions == 2:

            # Alpha missing, so calculate it from the dcm.
            if ("dcm_ab" in aero_parameter_names) and ("alpha" not in aero_parameter_names):
                transformation_code = "alpha = -rot_to_angle_2d(dcm_ab)"

            # Dcm missing, so calculate from alpha.
            elif ("alpha" in aero_parameter_names) and ("dcm_ab" not in aero_parameter_names):
                transformation_code = "dcm_ab = angle_to_rot_2d(-alpha)"

            # Both available, so do nothing
            elif {'alpha', 'dcm_ab'} <= aero_parameter_names:
                pass

            # Both missing, check if we have default_alpha.
            elif default_alpha is not None:
                transformation_code = \
                    f"alpha = {default_alpha}\n" \
                    "dcm_ab = angle_to_rot_2d(-alpha)"

            # If both missing and no default_alpha, raise an error.
            else:
                raise AttributeError(
                    f"2d aerodynamic calculations require either 'alpha' or 'dcm_ab' to be input parameters.")

            # Either we have raised the AttributeError or we have both alpha and dcm_ab. So add alpha to the set
            # and discard the dcm_ab because the coefficient models do not support it.
            aero_parameter_names.add('alpha')
            aero_parameter_names.discard('dcm_ab')

        # For the 3d case we need dcm_ab, alpha and beta.
        elif output_dimensions == 3:

            # We have everything we need, so do nothing.
            if {'alpha', 'beta', 'dcm_ab'} <= aero_parameter_names:
                pass

            # Calculate alpha and beta from the dcm.
            elif "dcm_ab" in aero_parameter_names:
                transformation_code = "_, minus_alpha, beta = dcm_to_euler(dcm_ab_x, 'xyz'); alpha = -minus_alpha"

            # Calculate the dcm from alpha and beta.
            elif {'alpha', 'beta'} <= aero_parameter_names:
                transformation_code = "dcm_ab = tr_ab(alpha, beta)"

            # Calculate the dcm from alpha and default_beta.
            elif "alpha" in aero_parameter_names and (default_beta is not None):
                transformation_code = \
                    f"beta = {default_beta}\n" \
                    f"dcm_ab = tr_ab(alpha, beta)"

            # Calculate the dcm from beta and default_alpha.
            elif "beta" in aero_parameter_names and (default_alpha is not None):
                transformation_code = \
                    f"alpha = {default_alpha}\n" \
                    f"dcm_ab = tr_ab(alpha, beta)"

            # Calculate the dcm from default_alpha and default_beta.
            elif (default_alpha is not None) and (default_beta is not None):
                transformation_code = \
                    f"alpha = {default_alpha}\n" \
                    f"beta = {default_beta}\n" \
                    f"dcm_ab = tr_ab(alpha, beta)"

            # We have none of them and no default values. We can't do anything, so throw an error.
            else:
                raise AttributeError(
                    f"3d aerodynamic calculations require either 'alpha' and 'beta' or 'dcm_ab' to be input parameters.")

            # Either we have raised the error or we have all three (alpha, beta, dcm_ab), so add alpha and
            # beta. And remove dcm_ab, because it's not supported as an input parameter by the coefficient models.
            aero_parameter_names.add('alpha')
            aero_parameter_names.add('beta')
            aero_parameter_names.discard('dcm_ab')

        # Add the transformation code to the coefficient calculation code builder.
        if transformation_code is not None:
            force_moment_calculation_code_builder.write(
                "# Calculate dcm_ab or alpha/beta, depending on what is missing and number of dimension.\n"
                f"{transformation_code}\n\n")

        # Initialize the coefficient code dictionary, with either zero values if a
        # required coefficient model is missing or None if the code will have to be
        # generated later.
        if output_dimensions == 1:
            # For a 1d you either need the c_x or c_d force coefficients.
            if "c_x" in self.coefficients:
                coefficients_code_dict['c_x'] = None

            elif "c_d" in self.coefficients:
                coefficients_code_dict['c_d'] = None
                in_aerodynamic_reference_frame = True

                # If neither was given then set c_x to zero. This would mean no aerodynamics.
            else:
                coefficients_code_dict['c_x'] = "0.0"

        elif output_dimensions == 2:
            # For a 2d you need c_m and either (c_x, c_z) or (c_d, c_l)

            coefficients_code_dict['c_m'] = None if "c_m" in self.coefficients else "0.0"

            if ("c_x" in self.coefficients) or ("c_z" in self.coefficients):
                coefficients_code_dict['c_x'] = None if "c_x" in self.coefficients else "0.0"
                coefficients_code_dict['c_z'] = None if "c_z" in self.coefficients else "0.0"
            elif ("c_d" in self.coefficients) or ("c_l" in self.coefficients):
                coefficients_code_dict['c_d'] = None if "c_d" in self.coefficients else "0.0"
                coefficients_code_dict['c_l'] = None if "c_l" in self.coefficients else "0.0"
                in_aerodynamic_reference_frame = True
            else:
                coefficients_code_dict['c_x'] = "0.0"
                coefficients_code_dict['c_z'] = "0.0"

        else:  # 3d
            # For 3d you need c_ll_, c_m and c_ln moment coefficients.
            # And either (c_x, c_y, c_z) or (c_d, c_q, c_l).

            coefficients_code_dict['c_ll'] = None if "c_ll" in self.coefficients else "0.0"
            coefficients_code_dict['c_m'] = None if "c_m" in self.coefficients else "0.0"
            coefficients_code_dict['c_ln'] = None if "c_ln" in self.coefficients else "0.0"

            if ("c_x" in self.coefficients) or ("c_y" in self.coefficients) or ("c_z" in self.coefficients):
                coefficients_code_dict['c_x'] = None if "c_x" in self.coefficients else "0.0"
                coefficients_code_dict['c_y'] = None if "c_y" in self.coefficients else "0.0"
                coefficients_code_dict['c_z'] = None if "c_z" in self.coefficients else "0.0"
            elif ("c_d" in self.coefficients) or ("c_q" in self.coefficients) or ("c_l" in self.coefficients):
                coefficients_code_dict['c_d'] = None if "c_d" in self.coefficients else "0.0"
                coefficients_code_dict['c_q'] = None if "c_q" in self.coefficients else "0.0"
                coefficients_code_dict['c_l'] = None if "c_l" in self.coefficients else "0.0"
                in_aerodynamic_reference_frame = True
            else:
                coefficients_code_dict['c_x'] = "0.0"
                coefficients_code_dict['c_y'] = "0.0"
                coefficients_code_dict['c_z'] = "0.0"

        # Loop through the coefficient code dictionary to generate the code for the
        # the coefficients who's models we have and print a warning for the coefficients
        # we don't have models for.
        for coef_name, value in coefficients_code_dict.items():

            # If the value is None, then we need the generate the code for the coefficient,
            # otherwise it's already been set to 0.0 because we don't have the model for this
            # coefficient.
            if value is None:
                # Get coefficient model instance.
                coef_model = self.coefficients[coef_name]

                # Get a set with the input parameters of the coefficient model.
                coef_parameter_names = set(coef_model.parameter_names)

                # Check if the coefficient model is missing one or more input parameters that the
                # aerodynamic model has.
                only_aero_model = aero_parameter_names - coef_parameter_names
                if len(only_aero_model) > 0:

                    # If the coefficient model only has mach as an input parameter, then
                    # it's a point drag/lift model (e.g. simple_parachute).
                    if coef_parameter_names == {"mach"} and (coef_name in ['c_d', 'c_l', 'c_q']):
                        pass

                    # NOTE: We could also support the 3d case where the coef model has mach and alpha, but
                    # no beta. In this case, it could be assumed that the alpha of the coef model the
                    # total angle of attack and in the end the coefficients would have to be rotated with
                    # phi, but this requires exceptions to be added in quite a few places.

                    # Any other situation is not supported.
                    else:
                        raise AttributeError(
                            f"The '{coef_name}' coefficient model is missing the aerodynamic input parameters '{only_aero_model}'")

                # Initialize coefficient string with the function call to the get_coefficient of the coef_model.
                coefficient_code = f"self.coefficients['{coef_name}'].get_coefficient("

                # Loop through the coefficient input parameters.
                for coef_parameter_name in sorted(coef_parameter_names):
                    if coef_parameter_name in aero_parameter_names:
                        # If the aerodynamic model has the input parameter, pass it.
                        coefficient_code += f"{coef_parameter_name}={coef_parameter_name}, "
                    else:
                        # If not, find the default value for the input parameter and pass it.
                        coefficient_code += \
                            f"""{coef_parameter_name}={default_values[coef_parameter_name] or coef_model.default_values[coef_parameter_name]}, """

                # Close the function call bracket and add it to the coefficients code dictionary.
                coefficient_code += ")"
                coefficients_code_dict[coef_name] = coefficient_code

            else:
                # If the aerodynamic model is missing the coefficient, print a warning.
                print(f"WARNING: Aerodynamic model is missing required coefficient '{coef_name}'. Setting it to '0.0'.")

        # Write the coefficient model calls to the force moment calculation code builder.
        force_moment_calculation_code_builder.write(
            "# Get the values of the aerodynamic coefficients from the coefficient models.\n"
            "# If the model for a coefficient is missing, set it to '0.0'.\n")

        for coef_name, coef_code in coefficients_code_dict.items():
            force_moment_calculation_code_builder.write(f"{coef_name} = {coef_code}\n")

        force_moment_calculation_code_builder.write("\n")

        # Generate the code for the force and moment coefficient vectors.
        # c_mom_x0 = Moment coefficient at the moment reference center (M.R.C.)
        if in_aerodynamic_reference_frame:
            if output_dimensions == 1:
                force_coefficient_vector = "c_force = -c_d"
            elif output_dimensions == 2:
                force_coefficient_vector = "c_force = dcm_ab.T @ np.array([[-c_d], [-c_l]])"
            elif output_dimensions == 3:
                force_coefficient_vector = "c_force = dcm_ab.T @ np.array([[-c_d], [-c_q], [-c_l]])"
        else:
            if output_dimensions == 1:
                force_coefficient_vector = "c_force = c_x"
            elif output_dimensions == 2:
                force_coefficient_vector = "c_force = np.array([c_x, 0, c_z])"
            elif output_dimensions == 3:
                force_coefficient_vector = "c_force = np.array([c_x, c_y, c_z])"

        if output_dimensions == 1:
            moment_coefficient_vector = "c_mom = 0.0"
        elif output_dimensions == 2:
            moment_coefficient_vector = \
                "cg_3 = np.array([cg[0, 0], 0, cg[1, 0]])\n" \
                "c_mom_x0 = np.array([0, c_m, 0])\n" \
                "c_mom = (c_mom_x0 + np.cross(self.mrc.reshape((3,)) - cg_3, c_force) / self.lref)[1]\n" \
                "c_force = np.array([[c_x], [c_z]])"
        elif output_dimensions == 3:
            moment_coefficient_vector = \
                "cg_3 = cg.reshape((3,))\n" \
                "c_mom_x0 = np.array([c_ll, c_m, c_ln])\n" \
                "c_mom = c_mom_x0 + (np.cross(self.mrc.reshape((3,)) - cg_3, c_force) / self.lref)\n" \
                "c_mom = c_mom.reshape((3, 1))\n" \
                "c_force = c_force.reshape((3, 1))"

        force_moment_calculation_code_builder.write(
            "# Calculate the force and moment coefficient vectors in body axes. \n"
            f"{force_coefficient_vector}\n\n{moment_coefficient_vector}\n\n")

        # Calculate the aerodynamic forces and moments.
        # force_moment_calculation_code_builder.write(
        #     "# Calculate the aerodynamic forces and moments.\n"
        #     "v_a_abs = np.linalg.norm(v_a)\n"
        #     "f = 0.5 * rho * v_a_abs**2 * self.sref * c_force\n"
        #     "m = 0.5 * rho * v_a_abs**2 * self.sref * self.lref * c_mom\n")

        # Get a string containing the code and compile it into a code object.
        self.code_str = force_moment_calculation_code_builder.getvalue()

        # print_code(self.code_str)

        self.code_obj = compile(self.code_str, f"<generated_aerodynamic_model_for={self.path}>", "exec")

        self.aero_parameter_names = aero_parameter_names

        return self.code_str


def abs_path(rel_path):
    """
    Returns an absolute path to a file relative to this file.
    :param rel_path: Path relative to this file
    :return: Absolute path
    """
    return os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), rel_path))

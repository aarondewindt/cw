import unittest
from pprint import PrettyPrinter
from hashlib import md5

import numpy as np
from numpy import testing as npt
import matplotlib.pyplot as plt
from dateutil.parser import parse as datetime_parse

from cw.aero_file import AeroFile
from cw.aero_file.regular_grid_interpolation_coefficient_model import RegularGridInterpolationCoefficientModel
from cw.aero_file.linear_nd_interpolation_coefficient_model import LinearNDInterpolationCoefficientModel
from cw.flex_file import flex_load
from cw.special_print import code_print
from cw.test import test_path


# Create a pretty printer
pp = PrettyPrinter(indent=2)
pprint = pp.pprint


class TestAeroFile(unittest.TestCase):
    def test_load_file(self):
        """
        Loads in a file and checks whether all expected coefficients where loaded
        with the correct model.
        """
        aero = AeroFile(test_path / "data_files" / "aero_1.yaml")

        # Check if the correct aero model was loaded in.
        self.assertIsInstance(aero.coefficients['c_d'], RegularGridInterpolationCoefficientModel,
                              "Wrong type for the cd aerodynamic model.")

        self.assertIsInstance(aero.coefficients['c_x'], RegularGridInterpolationCoefficientModel,
                              "Wrong type for the ca aerodynamic model.")

        self.assertIsInstance(aero.coefficients['c_m'], LinearNDInterpolationCoefficientModel,
                              "Wrong type for the cm aerodynamic model.")

    def test_load_from_old_file_no_deflections(self):
        """
        Loads in an old data file, while discarding the deflection information.
        """
        aero = AeroFile.from_old_format(test_path / "data_files" / "a.p")

        c_names = ["c_x", "c_y", "c_z", "c_ll", "c_m", "c_ln"]

        self.assertCountEqual(c_names,
                              aero.coefficients.keys())

        for c_name in c_names:
            with self.subTest(c_name=c_name):
                self.assertIsInstance(aero.coefficients[c_name], RegularGridInterpolationCoefficientModel)

                # Check that these parameters where read in correctly.
                self.assertEqual(
                    [-14, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -0.1,
                     0, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
                    aero.coefficients[c_name].parameters['alpha'],
                )

                self.assertEqual(
                    [-14, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
                    aero.coefficients[c_name].parameters['beta'],
                )

                self.assertEqual(
                    [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
                    aero.coefficients[c_name].parameters['mach'],
                )

                # Make sure that these parameters where removed from the data.
                self.assertNotIn('deflctP', aero.coefficients[c_name].parameters)
                self.assertNotIn('deflctQ', aero.coefficients[c_name].parameters)
                self.assertNotIn('deflctR', aero.coefficients[c_name].parameters)

                # Check the shape of the data table.
                self.assertEqual((27, 25, 8), aero.coefficients[c_name].table.shape)

    def test_load_from_old_file_with_deflections(self):
        """
        Loads in an old aerodynamic file, while keeping the deflection information.
        """
        aero = AeroFile.from_old_format(test_path / "data_files" / "a.p", no_deflections=False)

        c_names = ["c_x", "c_y", "c_z", "c_ll", "c_m", "c_ln"]

        self.assertCountEqual(c_names,
                              aero.coefficients.keys())

        for c_name in c_names:
            with self.subTest(c_name=c_name):
                self.assertIsInstance(aero.coefficients[c_name], RegularGridInterpolationCoefficientModel)

                # Check that these parameters where read in correctly.
                self.assertEqual(
                    [-14, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -0.1, 0, 0.1, 1, 2, 3, 4, 5, 6, 7,
                     8, 9, 10, 12, 14],
                    aero.coefficients[c_name].parameters['alpha'],
                )

                self.assertEqual(
                    [-14, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     12, 14],
                    aero.coefficients[c_name].parameters['beta'],
                )

                self.assertEqual(
                    [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
                    aero.coefficients[c_name].parameters['mach'],
                )

                self.assertEqual(
                    [0],
                    aero.coefficients[c_name].parameters['deflctP'],
                )

                self.assertEqual(
                    [0],
                    aero.coefficients[c_name].parameters['deflctQ'],
                )

                self.assertEqual(
                    [0],
                    aero.coefficients[c_name].parameters['deflctR'],
                )

                # Check the shape of the data table.
                self.assertEqual((27, 25, 8, 1, 1, 1), aero.coefficients[c_name].table.shape)

    def test_regular_grid_interpolation(self):
        """
        Does different test to check whether the regular_grid_interpolation model
        works properly.
        """
        aero = AeroFile(test_path / "data_files" / "aero_1.yaml")

        with self.subTest(msg="1D table"):
            # check if the table was loaded in correctly
            npt.assert_allclose([0.5, 0.4, 0.6, 0.6, 0.5, 0.45], aero.coefficients['c_d'].table,
                                err_msg="Coefficient table loaded incorrectly.")

            self.assertEqual([0, 0.4, 0.8, 1.2, 2, 6], aero.coefficients['c_d'].parameters['mach'],
                             "Mach number loaded incorrectly.")

            # Check if the interpolator works.
            self.assertEqual(0.45, aero.coefficients['c_d'].get_coefficient(mach=0.2))

            # Check the data dump
            self.assertDictEqual({
                'table': [0.5, 0.4, 0.6, 0.6, 0.5, 0.45],
                'method': 'linear',
                'bounds_error': False,
                'fill_value': np.nan,
                'parameters':
                    [
                        ['mach', [0, 0.4, 0.8, 1.2, 2, 6]]
                    ]
            }, aero.coefficients['c_d'].dump_data())

        with self.subTest(msg="2D table"):
            self.assertAlmostEqual(aero.coefficients['c_x'].get_coefficient(1, 0.3), -1.0484114285714288)
            self.assertDictEqual({
                'table': [[-1.2577, -1.030525, -0.89535, -0.81665, -0.7645],
                          [-1.2977, -1.06395, -0.924775, -0.84371667, -0.79005],
                          [-1.3065, -1.071325, -0.9312, -0.84965833, -0.7957],
                          [-1.2977, -1.06395, -0.924775, -0.84371667, -0.79005],
                          [-1.2577, -1.030525, -0.89535, -0.81665, -0.7645]],
                'method': 'linear',
                'bounds_error': False,
                'fill_value': np.nan,
                'parameters': [
                    ['alpha', [-5.0, -2.5, 0.0, 2.5, 5.0]],
                    ['mach', [0.1, 0.275, 0.45, 0.625, 0.8]],
                ]
            }, aero.coefficients['c_x'].dump_data())

    def test_regular_grid_interpolation_point_value_tables(self):
        """
        Tests the 'point_value_tables()' function of the regular grid interpolator model.
        """
        aero = AeroFile(test_path / "data_files" / "aero_1.yaml")
        x, y = aero.coefficients['c_x'].point_value_tables()

        correct_x = [
            [-5., 0.1],
            [-5., 0.275],
            [-5., 0.45],
            [-5., 0.625],
            [-5., 0.8],
            [-2.5, 0.1],
            [-2.5, 0.275],
            [-2.5, 0.45],
            [-2.5, 0.625],
            [-2.5, 0.8],
            [0., 0.1],
            [0., 0.275],
            [0., 0.45],
            [0., 0.625],
            [0., 0.8],
            [2.5, 0.1],
            [2.5, 0.275],
            [2.5, 0.45],
            [2.5, 0.625],
            [2.5, 0.8],
            [5., 0.1],
            [5., 0.275],
            [5., 0.45],
            [5., 0.625],
            [5., 0.8]]

        correct_y = [
            -1.2577, -1.030525, -0.89535, -0.81665, -0.7645,
            -1.2977, -1.06395, -0.924775, -0.84371667, -0.79005,
            -1.3065, -1.071325, -0.9312, -0.84965833, -0.7957,
            -1.2977, -1.06395, -0.924775, -0.84371667, -0.79005,
            -1.2577, -1.030525, -0.89535, -0.81665, -0.7645]

        npt.assert_allclose(x, correct_x)
        npt.assert_allclose(y, correct_y)

    def test_linear_nd_interpolation(self):
        """
        Does different test to check whether the linear_nd_interpolation model
        works properly.
        """
        aero = AeroFile(test_path / "data_files" / "aero_1.yaml")

        correct_points = [
            [-1.2, 0.1], [0.0, 0.1], [4.0, 0.1], [8.0, 0.1], [12.0, 0.1],
            [-1.2, 0.5], [0.0, 0.5], [4.0, 0.5], [8.0, 0.5], [12.0, 0.5],
            [-1.2, 0.9], [0.0, 0.9], [4.0, 0.9], [8.0, 0.9], [12.0, 0.9],
            [-1.2, 1.3], [0.0, 1.3], [4.0, 1.3], [8.0, 1.3], [12.0, 1.3],
            [-1.2, 1.7], [0.0, 1.7], [4.0, 1.7], [8.0, 1.7], [12.0, 1.7]
        ]

        correct_values = [0.2513, 0.0, -2.4088, -9.3068, -20.6336, 0.2527, 0.0, -2.4603,
                          -9.7158, -21.9973, 0.2545, 0.0, -2.5167, -10.1453, -23.4089,
                          0.3008, 0.0, -3.0652, -12.6461, -31.5364, 0.3055, 0.0, -3.1374,
                          -13.5819, -35.3548]

        with self.subTest(msg="get_coefficient"):
            self.assertAlmostEqual(aero.coefficients['c_m'].get_coefficient(1, 0.3), -0.615075)

        with self.subTest(msg="data_dump"):
            data_dump = aero.coefficients['c_m'].dump_data()
            self.assertEqual(data_dump['parameter_names'], ['alpha', 'mach'])
            npt.assert_allclose(data_dump['points'], correct_points)

            npt.assert_allclose(data_dump['values'], correct_values)

            self.assertTrue(np.isnan(data_dump['fill_value']))
            self.assertEqual(data_dump['rescale'], False)

        with self.subTest(msg="point_value_tables"):
            x, y = aero.coefficients['c_m'].point_value_tables()
            npt.assert_allclose(x, correct_points)
            npt.assert_allclose(y, correct_values)

    def test_dump(self):
        aero_file = AeroFile(test_path / "data_files" / "aero_1.yaml")
        aero_file.dump(test_path / "data_files" / "aero_2.i.yaml")

        aero_1_data = flex_load(test_path / "data_files" / "aero_1.yaml")
        aero_2_data = flex_load(test_path / "data_files" / "aero_2.i.yaml")

        # Check whether the data in the aero_2_data dictionary is the same as in aero_1_data.
        # Since aero_1_data was written by hand, it doesn't contain all the optional parameters
        # for the coefficient models, while aero_2_data does. So we are only checking the parameters
        # in aero_1_data and ignore any parameter unique to aero_2_data.

        def check_dictionaries(d1, d2):
            for key, value in d1.items():
                # print(f"dict key='{key}, v1='{value}, v2={d2[key]}")
                if isinstance(value, dict):
                    self.assertIsInstance(d2[key], dict)
                    check_dictionaries(value, d2[key])
                elif isinstance(value, float):
                    self.assertAlmostEqual(value, d2[key])
                elif isinstance(value, (list, tuple)):
                    self.assertIsInstance(d2[key], (list, tuple))
                    check_list(value, d2[key])
                elif isinstance(value, set):
                    # It's assumed that sets do not contain lists, dictionaries or
                    # other sets and that both sets should contain the same elements.
                    self.assertIsInstance(d2[key], set)
                    self.assertEqual(len(d2[key] - value), 0)
                else:
                    self.assertEqual(value, d2[key])

        def check_list(l1, l2):
            for i, value in enumerate(l1):
                # print(f"list i='{i}, v1='{value}, v2={l2[i]}")
                if isinstance(value, dict):
                    self.assertIsInstance(l2[i], dict)
                    check_dictionaries(value, l2[i])
                elif isinstance(value, float):
                    self.assertAlmostEqual(value, l2[i])
                elif isinstance(value, (list, tuple)):
                    self.assertIsInstance(l2[i], (list, tuple))
                    check_list(value, l2[i])
                elif isinstance(value, set):
                    # It's assumed that sets do not contain lists, dictionaries or
                    # other sets and that both sets should contain the same elements.
                    self.assertIsInstance(l2[i], set)
                    self.assertEqual(len(l2[i] - value), 0)
                else:
                    self.assertEqual(value, l2[i])

        aero_2_data['created'] = datetime_parse(aero_2_data['created'])

        check_dictionaries(aero_1_data, aero_2_data)

    def test_setup_aerodynamic_forces_moments_calculation_simple(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")

        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha"},
            output_dimensions=3,
        )

        print("")
        # print(code)
        code_print(code)

    def test_forces_moments_coefficients(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")

        c_fs = []
        alphas = list(range(-10, 11))

        for alpha in alphas:
            c_fs.append(aero_file.calc_forces_moments_coefficients(alpha=alpha, output_dimensions=3)[0])

        plt.plot(alphas, np.squeeze(c_fs))
        plt.show()

    def test_setup_aerodynamic_forces_moments_calculation_parachute_1d(self):
        aero_file = AeroFile(test_path / "data_files" / "aero_para_simple.yaml")

        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach"},
            output_dimensions=2,
            # default_alpha=1.0
        )

        print("test_setup_aerodynamic_forces_moments_calculation_parachute_1d")
        print(f"'''{code}'''")
        # code_print(code)

        md5_obj = md5()
        md5_obj.update(code.encode("utf-8"))
        hash = md5_obj.hexdigest()
        self.assertEqual(hash, "ff4d6f485e1c0e42b662e57f082ce907")

    def test_setup_aerodynamic_forces_moments_calculation_parachute_2d(self):
        aero_file = AeroFile(test_path / "data_files" / "aero_para_simple.yaml")

        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach"},
            output_dimensions=2,
            # default_alpha=1.0
        )

        print("test_setup_aerodynamic_forces_moments_calculation_parachute_2d")
        code_print(code)

        md5_obj = md5()
        md5_obj.update(code.encode("utf-8"))
        hash = md5_obj.hexdigest()
        self.assertEqual(hash, "ff4d6f485e1c0e42b662e57f082ce907")

    def test_setup_aerodynamic_forces_moments_calculation_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "aero_para_simple.yaml")

        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach"},
            output_dimensions=3,
            # default_alpha=1.0
        )

        print("test_setup_aerodynamic_forces_moments_calculation_parachute_3d")
        code_print(code)

        md5_obj = md5()
        md5_obj.update(code.encode("utf-8"))
        hash = md5_obj.hexdigest()
        self.assertEqual(hash, "da4b66f5119ad6a87aeb3a2fed675fd1")

    def test_invalid_dimension(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")

        for i in range(4, 8):
            with self.subTest(k=i):
                with self.assertRaisesRegex(ValueError, f"'{i}' is an invalid value for 'output_dimensions'\."):
                    aero_file.setup_aerodynamic_forces_moments_calculation({'alpha', 'mach', 'beta'}, i)


class TestSetupAerodynamicForcesMomentCalculation(unittest.TestCase):
    def assert_md5(self, code, correct_hash):
        md5_obj = md5()
        md5_obj.update(code.encode("utf-8"))
        hash = md5_obj.hexdigest()
        self.assertEqual(correct_hash, hash)

    def test_alpha_missing_parachute_2d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"dcm_ab", "beta", "mach"},
            output_dimensions=2
        )

        # print("test_alpha_missing")
        # code_print(code)

        self.assert_md5(code, "3ee941c7cb34d83af6acced41038fa50")

    def test_dcm_missing_parachute_2d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach"},
            output_dimensions=2
        )

        # print("test_dcm_missing")
        # code_print(code)
        self.assert_md5(code, "07da3fc6f5c192cd91eb0da7dc6cd5e8")

    def test_alpha_dcm_available_parachute_2d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach", "dcm_ab"},
            output_dimensions=2
        )

        # print("test_alpha_dcm_available")
        # code_print(code)
        self.assert_md5(code, "995ce2376db78884bc4ecf7106792ae3")

    def test_alpha_dcm_missing_use_defeault_alpha_parachute_2d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"beta", "mach"},
            output_dimensions=2
        )

        # print("test_alpha_dcm_missing")
        # code_print(code)
        self.assert_md5(code, "0aa6f2fe6e787a56aab2ae47898186b1")

    def test_alpha_dcm_missing_default_alpha_missing_parachute_2d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")

        with self.assertRaisesRegex(AttributeError, r"2d aerodynamic calculations require either 'alpha' or 'dcm_ab' to be input parameters\."):
            aero_file.setup_aerodynamic_forces_moments_calculation(
                input_parameters={"beta", "mach"},
                output_dimensions=2,
                default_alpha=None
            )

    def test_discard_dcm_parachute_2d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach", "dcm_ab"},
            output_dimensions=2
        )

        # print("test_discard_dcm")
        # code_print(code)

        self.assertEqual(aero_file.aero_parameter_names, {"alpha", "beta", "mach"})

    def test_nothing_missing_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach", "dcm_ab"},
            output_dimensions=3
        )

        # print("test_nothing_missing")
        # code_print(code)
        self.assert_md5(code, "c3d3e420bee34e1f240c7865ca08b1a2")

    def test_alpha_and_beta_missing_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"mach", "dcm_ab"},
            output_dimensions=3
        )

        # print("test_alpha_beta_missing")
        # code_print(code)
        self.assert_md5(code, "4fc7d5906af7f7e5734d413fd1665b3c")

    def test_dcm_missing_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"mach", "alpha", "beta"},
            output_dimensions=3
        )

        # print("test_dcm_missing")
        # code_print(code)
        self.assert_md5(code, "966128c6c4c27d523baa9fa22721a233")

    def test_calculate_dcm_from_alpha_and_default_beta_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "mach"},
            output_dimensions=3
        )

        # print("test_calculate_dcm_from_alpha_default_beta_3d")
        # code_print(code)
        self.assert_md5(code, "1be66bf6b65e1ef7646ff1d75f362323")

    def test_calculate_dcm_from_default_alpha_and_beta_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"beta", "mach"},
            output_dimensions=3
        )

        # print("test_calculate_dcm_from_default_alpha_and_beta_3d")
        # code_print(code)
        self.assert_md5(code, "fa32af0ac4240f54ba57cae075027cfb")

    def test_calculate_dcm_from_default_alpha_and_default_beta_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"mach"},
            output_dimensions=3
        )

        # print("test_calculate_dcm_from_default_alpha_and_default_beta_3d")
        # code_print(code)
        self.assert_md5(code, "374ffe1ba1b6f65df8a745fbd892956a")

    def test_everything_missing_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")

        with self.assertRaises(AttributeError):
            aero_file.setup_aerodynamic_forces_moments_calculation(
                input_parameters={"mach"},
                output_dimensions=3,
                default_alpha=None
            )

    def test_discard_dcm_parachute_3d(self):
        aero_file = AeroFile(test_path / "data_files" / "3d_aero.aero")
        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach", "dcm_ab"},
            output_dimensions=3
        )

        self.assertEqual(aero_file.aero_parameter_names, {"alpha", "beta", "mach"})

    def test_no_aerodynamics_parachute_1d(self):
        aero_file = AeroFile(test_path / "data_files" / "no_aero_dynamics_parachute_1D.yaml")

        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"mach"},
            output_dimensions=1
        )

        # print("test_no_aerodynamics_parachute_1D")
        # print("test_no_aerodynamics_parachute_1D")
        # print(code)
        # code_print(code)
        self.assert_md5(code, "8255a47a2a62c10f16a45426a0d4fd01")

    def test_coefficient_model_is_missing_aerodynamic_input_parameters_parachute_1d(self):
        aero_file = AeroFile(test_path / "data_files" / "Missing_aerodynamic_input_1D.yaml")

        with self.assertRaises(AttributeError):
            aero_file.setup_aerodynamic_forces_moments_calculation(
                input_parameters={"mach"},
                output_dimensions=1
            )

    def test_aerodynamic_model_missing_required_coefficient(self):
        aero_file = AeroFile(test_path / "data_files" / "aero_para_simple.yaml")

        code = aero_file.setup_aerodynamic_forces_moments_calculation(
            input_parameters={"alpha", "beta", "mach"},
            output_dimensions=2
        )

        # print("")
        # print(code)
        # code_print(code)
        self.assert_md5(code, "ff4d6f485e1c0e42b662e57f082ce907")


def hash(thing):
    """
    Uses md5 to create a hash of the thing and returns a hex string of the hash.
    It's useful for debugging and testing things.

    :param thing: The thing
    :return: Hash
    :rtype: str
    """
    md5hash = md5()
    md5hash.update(thing)
    return md5hash.hexdigest()

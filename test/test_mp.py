import unittest
from pathlib import Path
import pickle

import pandas as pd

from test import test_path

from cw.mp import Project
from cw.rm import rrmdir
from cw.mp import BatchConfigurationBase, run_project_locally

test_temp_projects_path = test_path / "mp.i.temp_projects"
test_temp_projects_path.mkdir(exist_ok=True)
test_projects_path = test_path / "mp_projects"
test_projects_path.mkdir(exist_ok=True)


class TestMP(unittest.TestCase):
    def test_find_project_root_path(self):
        """
        Tests whether the find_batch_root function works as expected.
        """
        project_path: Path = test_temp_projects_path / "root_find"
        rrmdir(project_path)
        project_path.mkdir()

        # The dir still doesn't contain the .cwmp dir. So it should raise an error.
        with self.assertRaisesRegex(
                ValueError,
                r"Not a valid cw.mp project directory \(or any of the parent directories\): Missing \.cwmp directory"):
            Project.find_package_root_path(project_path)

        # We now create the .dcds dir and try to find it.
        (project_path / ".cwmp").mkdir()
        root_dir = Project.find_package_root_path(project_path)
        self.assertTrue(root_dir.samefile(project_path))

        # Now from a subdirectory
        subdir = project_path / "subdir"
        subdir.mkdir()
        root_dir = Project.find_package_root_path(subdir)
        self.assertTrue(root_dir.samefile(project_path))

    def test_initialize_new_project_directory(self):
        """
        Initializes a new cw.mp project and checks whether it contains the correct files.
        """
        project_path = test_temp_projects_path / "init_batch"
        rrmdir(project_path)
        Project.initialize(project_path)

        # List of all files and directories whitch should have been created.
        correct_paths = {'init_batch', '.cwmp', 'readme.md', 'init_batch/configuration.py',
                         'init_batch/__init__.py', '.cwmp/cwmp.txt'}

        # Get list of all files and directories that where created.
        paths = set()
        for path in project_path.rglob("*"):
            paths.add(str(path.relative_to(project_path)))

        self.assertEqual(paths, correct_paths)
        
    def test_project_name(self):
        """
        Initializes a new directory and checks whether the find_source_package_name
        function works.
        """
        project_path = test_temp_projects_path / "package_name_batch"
        rrmdir(project_path)
        Project.initialize(project_path)

        package_name = Project.find_python_package_name(project_path)
        self.assertEqual("package_name_batch", package_name)

    def test_project_batch_configuration(self):
        """
        Initializes a new directory and check whether the configuration
        class is found correctly, by checking the instance the Batch class creates.
        """
        project_path = test_temp_projects_path / "config_batch"
        project = Project.initialize(project_path)
        batch_instance = project.batch
        self.assertIsInstance(batch_instance, BatchConfigurationBase)
        rrmdir(project_path)

    def test_input_output_parameter_list(self):
        project_path = test_temp_projects_path / "inout_batch"
        project = Project.initialize(project_path)
        self.assertEqual(project.batch.input_parameters, ('in_example_param_1', 'in_example_param_2'))
        self.assertEqual(project.batch.output_parameters, ('out_example_param_1', 'out_example_param_2'))

    def test_local_pool(self):
        project_path = test_projects_path / "local_pool"
        project = Project(project_path)

        result = run_project_locally(project, "result.i", 8)
        for (row_idx, row_values), inputs in zip(result.iterrows(), project.batch.create_cases()):
            out_c = inputs.in_a + inputs.in_b
            out_d = inputs.in_a * inputs.in_b
            self.assertEqual(row_values['in_a'], inputs.in_a)
            self.assertEqual(row_values['in_b'], inputs.in_b)
            self.assertEqual(row_values['out_c'], out_c)
            self.assertEqual(row_values['out_d'], out_d)


import unittest
import os
from cw.directory_walk import directory_walk
import fnmatch


def abs_path(rel_path):
    """
    Returns an absolute path to a file relative to this file.

    :param rel_path: Path relative to this file
    :return: Absolute path
    """
    return os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), rel_path))


class TestDirectoryWalk(unittest.TestCase):
    def test_directory_walk(self):
        """
        Check whether the directory walk function works correctly without a filter.
        """

        # Walk through the directory and keep a list with the names of all files.
        file_names = []
        for file_path in directory_walk(abs_path("./data_files/test_hash_directory"),
                                        filter=lambda path: path.suffix not in ['.pyc']):
            file_names.append(file_path.name)

        correct_file_names = [
            'jas.qw', 'thrust.csv', 'blabla.json', 'something.dtsml', '__init__.py',
            'qwerty.csv', 'bksdj.txt', '.gitignore']

        # Check whether the list is correct.
        self.assertCountEqual(file_names, correct_file_names)

    def test_with_ignore_patterns(self):
        """
        Check whether the directory walk function works correctly with a filter
        :return:
        """

        # Create a filter function.
        ignore_patterns = ["*.qw", "*.pyc"]

        def ignore_if_matching(file_path):
            for pattern in ignore_patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    return False
            return True

        # Walk through the directory and keep a list with the names of all files.
        file_names = []
        for file_path in directory_walk(abs_path("./data_files/test_hash_directory"),
                                        ignore_if_matching):
            file_names.append(file_path.name)

        correct_file_names = [
            "thrust.csv",
            "blabla.json",
            "something.dtsml",
            "__init__.py",
            "qwerty.csv",
            "bksdj.txt",
            ".gitignore"
        ]

        # Check whether the list is correct.
        self.assertCountEqual(file_names, correct_file_names)

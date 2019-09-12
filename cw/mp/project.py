from pathlib import Path
from typing import Union
import os
from cw.cached import cached
import sys
import importlib

from cw.directory_checksum import DirectoryChecksum
from cw.mp.batch_configuration_base import BatchConfigurationBase


class Project:
    def __init__(self, path: Union[str, os.PathLike]):
        # Find the package path and add it to the python path.
        self.path: Path = self.find_package_root_path(path)
        sys.path.insert(0, str(self.path))

        # Find batch source package.
        self.source_package_name = self.find_python_package_name(self.path)

    @staticmethod
    def find_package_root_path(path: Union[str, Path]) -> Path:
        """
        Finds the path to the root directory of the batch. This is a directory containing
        a .cwmp directory. If the root directory is not found. A value error is raised.

        :param path: Path to the batch root or a sub directory
        :return: Path to the batch's root directory
        """
        # TODO: Check if this thing works on Windows.

        # Convert to a Path object if necessary and get the absolute path.
        path = path if isinstance(path, Path) else Path(path)
        path = path.absolute()

        # Loop until we have reached the file system root.
        while not path.samefile(path.root):
            if (path / ".cwmp").is_dir():
                # If the current path contains the .cwmp directory return it.
                return path
            else:
                # Otherwise change path to it's parent and loop again.
                path = path.parent

        # If we have reached this point, that means that we have loop all the way
        # up to the filesystem root. Thus the directory was not inside cwmp batch.
        raise ValueError("Not a valid cw.mp project directory (or any of the parent directories): Missing "
                         ".cwmp directory")

    @staticmethod
    def find_python_package_name(path: Path) -> str:
        """
        Finds the name of the directory containing the batch's source code. This will
        be subdirectory of path containing two files named "__init__.py" and "config.py".

        :param path: Path object to the batch root directory.
        :return: Package name
        """
        package_name = None
        # Find python package name of this batch. This will be the name of a
        # subdirectory containing two files named __init__.py and config.py
        for sub_path in path.glob("*"):
            if sub_path.is_dir():
                if (sub_path / "__init__.py").is_file() and (sub_path / "configuration.py").is_file():
                    package_name = sub_path.name
                    break

        if package_name is None:
            raise ValueError("Not a valid cw.mp batch directory (or any of the parent directories): Missing "
                             "source package.")

        return package_name

    @classmethod
    def is_package_directory(cls, path: Union[str, Path]) -> bool:
        """
        Checks whether a directory is a valid batch directory or subdirectory.Checked database

        :param path: Path to the directory to check.
        :return: True if valid, otherwise False
        """
        # Find batch root directory. If the batch root is not found the directory
        # is not a batch.
        try:
            path = cls.find_package_root_path(path)
        except ValueError:
            return False

        # More checks will be added here later.

        return True

    @classmethod
    def initialize(cls,
                   path: Union[str, Path],
                   batch_name: str= None,
                   parents: bool= False) -> "Project":
        """
        Initializes a new dcds batch at the given path. No errors are raised or changes are
        made if the directory already is a dcds batch.

        :param path: Root path of the new dcds batch.
        :param batch_name: Name of the batch. If None, the directory's name will be used.
        :param parents: If True, any missing parent directories are creates as needed.
        :return: Instance of the Batch class of the newly created batch.
        """

        # Make sure the path is an instance of the Path object.
        path = path if isinstance(path, Path) else Path(path)

        # Get a Path to the template files directory.
        templates_path: Path = Path(__file__).parent / "templates"

        # If no batch name was given use the directory name as the batch name.
        # Make sure the batch name is lower case with no spaces (spaces are replaced with underscores).
        batch_name = batch_name or path.name.lower().replace(" ", "_")

        # Create the new directory if it's missing.
        path.mkdir(exist_ok=True, parents=parents)

        # If the directory already is a dcds batch. Create and return an instance of the
        # Batch class.
        if cls.is_package_directory(path):
            return cls(path)

        # .cwmp directory
        cwmp_path = path / ".cwmp"
        cwmp_path.mkdir()

        # Empty file in .cwmp to make sure the directory is pushed by git.
        (cwmp_path / "cwmp.txt").write_text("Ignore this file.")

        # Create readme file
        (path / "readme.md").write_text(f"{batch_name}\n{'='*len(batch_name)}\n")

        # Create source directory. This directory will have the same name as the
        # batch.
        source_path = path / batch_name
        source_path.mkdir()

        # Create configuration file.
        (source_path / "configuration.py").write_text(
            (templates_path / "configuration.py").read_text().format(batch_name=batch_name)
        )

        # Create __init__.py file.
        (source_path / "__init__.py").write_text("")

        return cls(path)

    @cached
    def batch(self) -> BatchConfigurationBase:
        """
        Instance of the batch configuration.

        :return:
        """
        # Import the config module.
        module = importlib.import_module(f"{self.source_package_name}.configuration")

        # Find the configuration klass
        klass = None
        for member_name in dir(module):
            member = getattr(module, member_name)
            if isinstance(member, type):
                if issubclass(member, BatchConfigurationBase) and (member is not BatchConfigurationBase):
                    klass = member

        # Raise an error if the klass was not found.
        if klass is None:
            raise ValueError("Not a valid cwmp batch directory (or any of the parent directories): "
                             "Missing batch configuration.")

        # Return an instance of the klass.
        return klass()

    @cached
    def directory_checksum_validator(self) -> DirectoryChecksum:
        return DirectoryChecksum(self.path, checksum_file_path=self.path / ".cwmp" / "checksum")

    def create_checksum_file(self, force=False):
        return self.directory_checksum_validator.create_checksum_file(force=force)

    def delete_checksum_file(self):
        self.directory_checksum_validator.delete_checksum_file()

    def validate_directory(self, checksum_file_hash=None):
        return self.directory_checksum_validator.validate_directory(checksum_file_hash)

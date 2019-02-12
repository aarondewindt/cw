from pathlib import Path, PurePosixPath
import fnmatch
import hashlib
from cw.directory_walk import directory_walk
from io import StringIO
import os
import re

from typing import Union


class DirectoryChecksum:
    """
    Class used to generate a file containing the sha256 hashes of all files in a
    directory and its subdirectories. The instance of the class is also used to
    validate a directory using a preexisting hash file.

    :param dir_path: String or :class:`os.PathLike` object with the path to the target directory.
    :param ignore_patterns: Iterable with unix shell-style patterns of file and directory names
       to ignore. :func:`fnmatch.fnmatch` is used to match the patterns.
    :param: Path relative to the directory to the checksum file. If None it will be
    """

    def __init__(self, dir_path: Union[os.PathLike, str], ignore_patterns=None, checksum_file_path="./checksum"):
        self.dir_path = dir_path if isinstance(dir_path, Path) else Path(dir_path)
        """
        :class:`os.PathLike` object pointing to the target directory,
        """

        self.ignore_patterns = set(ignore_patterns or [])
        """
        Set of the patterns to ignore.
        """

        self.checksum_file_path = self.dir_path / checksum_file_path
        """
        Path object to the checksum file.
        """

    @property
    def has_checksum_file(self):
        """
        True if the directory already has a checksum file.
        """
        return self.checksum_file_path.is_file()

    def create_checksum_file(self, force=False):
        """
        Generate a new checksum file in the directory.
        :class:`sim_common.directory_checksum.ChecksumFileAlreadyExistsError` will be raised
        if the checksum.

        :param force: If True the checksum file will be (re)created whether it
           exists or not.
        :returns: A bytestring with the sha256 hash of the contents of the checksum file.
        """

        # Check if the checksum file has already been created and raise an error
        # if it has been, except if its forced.
        if self.has_checksum_file and not force:
            raise ChecksumFileAlreadyExistsError(self.dir_path)

        # Initialize dictionary containing the hashes of all files in the directory.
        dir_hashes = {}

        # Create hashing object and walk through all files in the directory and its
        # subdirectories.
        hash_generator = hashlib.sha256()
        for file_path in directory_walk(self.dir_path, self.ignore_pattern_filter):
            with open(file_path, "rb") as f:
                # Read the file chunk by chunk and update the hash generator.
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_generator.update(chunk)

                # Put the hash in the dictionary with the relative path of the file
                # w.r.t. the root directory as the key. The path will be a Posix path.
                dir_hashes[str(PurePosixPath(file_path.relative_to(self.dir_path)))] = hash_generator.hexdigest()

                # Create new hash generator.
                hash_generator = hashlib.sha256()

        # Initialize string io object used to generate the contents of the hash file.
        str_io = StringIO()

        # Write the file paths and hashes to the string io object. One
        # pair on each line separated by tabs.
        for file_path_str, file_hash in dir_hashes.items():
            str_io.write(f"{file_path_str}\t{file_hash}\n")

        # Generate the hash of the hash_file contents.
        hash_generator.update(str_io.getvalue().encode('utf-8'))

        # Save the hashes to the hash file.
        with open(self.checksum_file_path, "w") as f:
            f.write(str_io.getvalue())

        # Return the hash of the checksum file.
        return hash_generator.digest()

    def ignore_pattern_filter(self, path):
        """
        Checks whether a path doesn't match one of the ignore patterns.

        :param path: :class:`os.PathLike` object pointing a file or directory.
        :return: True if the path doesn't math any of the ignore patterns, otherwise False.
        """

        # Return False for the checksum file.
        if self.checksum_file_path.exists():
            if self.checksum_file_path.samefile(path):
                return False

        # Return True of no patterns where given.
        if self.ignore_patterns is None:
            return True

        # Loop through all patterns.
        for pattern in self.ignore_patterns:
            # Return False if the path matches an ignore pattern.
            if fnmatch.fnmatch(path.name, pattern):
                return False

        # If it reached this point, this means that it doesn't match any of the patterns,
        # so return True.
        return True

    def validate_directory(self, checksum_file_hash=None):
        """
        Validates a directory by checking the hash of all the files listed in the
        checksum file.

        :param checksum_file_hash: Optional bytestring with the hash of the sha256
           checksum file.
        :return: True if the directory content is valid.
        """

        # If the directory doesn't have a checksum file, raise an error.
        if not self.has_checksum_file:
            return False

        # Load in all the text in the checksum file.
        with open(self.checksum_file_path, "r") as f:
            checksum_file_content = f.read()

        # If a hash was given for the checksum file, check the validity
        # of the checksum file.
        if checksum_file_hash is not None:
            hash_generator = hashlib.sha256()
            hash_generator.update(checksum_file_content.encode("utf-8"))
            if checksum_file_hash != hash_generator.digest():
                return False

        # If we have reached this point and the checksum_file_hash was given, then we know that
        # the checksum file has not been corrupted. If the checksum_file_hash was not given then we
        # are not sure, so we need to check if there are any errors in the file.
        # This is done by adding and extra matching group to the regex "|(.)". This group will
        # match anything that has not been matched by the base regex. Meaning that
        # if something is matched by this group, there are syntax errors in the file.
        checksum_file_regex_pattern = \
            r"(?:(.+)(?:\t)([a-fA-F0-9]{64}))|(.)" if checksum_file_hash is None else r"(?:(.+)(?:\t)([a-fA-F0-9]{64}))"

        # Loop through all matches found by the regex.
        for match in re.finditer(checksum_file_regex_pattern, checksum_file_content):
            # Check if syntax error was found in case no checksum_file_hash was given.
            if checksum_file_hash is None:
                if match.group(3) is not None:
                    return False

            # Get the file path and file hash and convert these to the correct type.
            file_path = self.dir_path / match.group(1)
            file_hash = bytes.fromhex(match.group(2))

            # Generate the sha256 hash of the contents of the file.
            hash_generator = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_generator.update(chunk)

            # Return false if the hash doesn't match.
            if file_hash != hash_generator.digest():
                return False

        # If we have reached here, then all of the files have been checked and they are valid.
        # so return True.
        return True


class ChecksumFileAlreadyExistsError(Exception):
    """
    Raised when an attempt is being made to create a checksum file in a directory
    that already has one. If the intention is to recreate it, set the force parameter to True.
    """
    def __init__(self, dir_path):
        super().__init__(f"'{str(dir_path)}' already has a checksum file.")


class NoChecksumFileError(Exception):
    """
    Raised when an attempt is being made to validate a directory that doesn't have a
    checksum file.
    """
    def __init__(self, dir_path):
        super().__init__(f"'{str(dir_path)}' has no checksum file.")


class InvalidChecksumFile(Exception):
    """
    Raised when the directory has a invalid checksum file.
    """
    def __init__(self, dir_path):
        super().__init__(f"'{str(dir_path)}' has an invalid checksum file.")

from os import unlink, remove
from os.path import isdir, islink, exists
from shutil import rmtree
from pathlib import Path


def rm(path):
    """
    Deletes a file, directory or symlink.

    :param path: Path to file, directory or symlink to delete.
    """
    if exists(path):
        if isdir(path):
            if islink(path):
                unlink(path)
            else:
                rmtree(path)
        else:
            if islink(path):
                unlink(path)
            else:
                remove(path)


def rrmdir(dir_path: Path):
    """
    Deletes a directory and all of it's content.

    :param dir_path: Path object ot the directory.
    :return:
    """
    if dir_path.is_dir():
        for path in dir_path.rglob("*"):
            if path.is_dir():
                rrmdir(path)
            else:
                path.unlink()
        dir_path.rmdir()

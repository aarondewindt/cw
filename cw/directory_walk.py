from os import PathLike
from pathlib import Path


def directory_walk(dir_path, filter=None):
    """
    Generator that walks through a directory and all subdirectories and yields
    :class:`os.PathLike` instances to the files found.

    :param dir_path: String or :class:`os.PathLike` object with the path to the target directory.
    :param filter: Callable taking one :class:`os.PathLike` object as parameter. The callable
       should return True if the path should be yielded by the generator, otherwise False.
    """

    # If dir_path is not an os.PathLike object, try to convert it to one.
    dir_path = dir_path if isinstance(dir_path, PathLike) else Path(dir_path)

    # Iterate through all files and directories in the directory.
    for sub_path in dir_path.iterdir():

        # If a filter was given. Check whether the file or directory should be skipped.
        if filter is not None:
            if not filter(sub_path):
                continue

        if sub_path.is_dir():
            # If sub_path points to a directory yield from this function.
            # Yea I know that recursion could cause problems here if someone
            # decided to make a circular symlink.
            yield from directory_walk(sub_path, filter)
        else:
            # If it's anything else, yield it.
            yield sub_path

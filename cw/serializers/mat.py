"""
Module implementing a pickle like file serializer interface for matlab .mat file.
"""

__all__ = ["loads", "dumps", "load", "dump"]

from scipy.io import loadmat, savemat
from io import BytesIO

from typing import ByteString, Union, BinaryIO
from os import PathLike


def loads(bytes_object: Union[bytes, bytearray]) -> dict:
    """Read in a object hierarchy from a bytes object with the raw bytes of a .mat file."""
    # Scipy's can load object in from file like objects.
    # So we first write the byte array to BytesIO and then give it loadmat to load it in.
    bytes_io = BytesIO()
    bytes_io.write(bytes_object)
    obj = loadmat(bytes_io)

    # loadmat will add these extra items to the dictionary, so lets pop them.
    obj.pop("__globals__")
    obj.pop("__header__")
    obj.pop("__version__")
    return obj


def dumps(obj: dict) -> ByteString:
    """
    Serializes an object hierarchy into a matlab .mat file.

    :param obj: Object to serialize
    :return: Byte string with
    """
    bytes_object = BytesIO()
    savemat(bytes_object, obj)
    return bytes_object.getvalue()


def load(file: Union[str, BinaryIO, PathLike]) -> dict:
    """
    Load matlab .mat file.

    :param file: Path to the file or open file-like object ot dump to.
    :return: Loaded object.
    """
    file = str(file)
    obj = loadmat(file)
    obj.pop("__globals__")
    obj.pop("__header__")
    obj.pop("__version__")
    return obj


def dump(obj: dict, file: Union[str, BinaryIO, PathLike]):
    """
    Dumps an object into a matlab .mat file.

    :param obj: Object to dump
    :param file: Path to the file or open file-like object ot dump to.
    """
    file = str(file)
    savemat(file, obj)

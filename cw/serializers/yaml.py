"""
Module implementing a pickle like file serializer interface for .yaml files.
"""
# from io import BytesIO
# from typing import ByteString, Union, BinaryIO, Sequence
# from os import PathLike
import yaml
dumps = yaml.dump
loads = yaml.load
dump = yaml.dump
load = yaml.load

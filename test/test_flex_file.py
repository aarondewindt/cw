import unittest
import os
import sys
from traceback import format_exception
from pathlib import Path

from cw.flex_file import flex_load, flex_dump, determine_serialization
from cw.rm import rm, rrmdir
from cw.object_hierarchies import object_hierarchy_equals
from cw.serializers import yaml, msgpack
from test import test_path

from itertools import product
import pickle
import json


class TestFile(unittest.TestCase):
    def setUp(self):
        self.test_data = {
            "id": 1,
            "first_name": "Zebadiah",
            "last_name": "Anderton",
            "email": "zanderton0@bbc.co.uk",
            "gender": "Male",
            "ip_address": "160.42.195.192"
        }

        self.serializers = {
            "msgpack": msgpack,
            "pickle": pickle,
            "yaml": yaml,
            "json": json
        }

        self.file_extensions = {
            ".pickle": pickle,
            ".yaml": yaml,
            ".msgp": msgpack,
            ".json": json,
            ".unknown": None
        }

        # Clear the temporary directory to store the test files.
        self.flex_file_temp: Path = (test_path / "flex_file_temp")
        if self.flex_file_temp.exists():
            rrmdir(self.flex_file_temp)
        self.flex_file_temp.mkdir()

    def tearDown(self):
        pass
        # if self.flex_file_temp.exists():
        #     rrmdir(self.flex_file_temp)

    def test_dump_load(self):
        for (serializer_name, serializer), \
            (extension, extension_serializer), \
            is_gzipped,\
            is_gzipped_default in product(self.serializers.items(),
                                          self.file_extensions.items(),
                                          [True, False],
                                          [True, False]):
            with self.subTest(serializer=serializer_name,
                              extension=extension,
                              is_gzipped=is_gzipped,
                              is_gzipped_default=is_gzipped_default):
                file_path = self.flex_file_temp / f"data.i{extension}{'.gz' if is_gzipped else ''}"
                if extension == ".unknown":
                    is_actually_gzipped = is_gzipped or is_gzipped_default
                else:
                    is_actually_gzipped = is_gzipped

                flex_dump(self.test_data, file_path, default_serializer=serializer, default_is_gzipped=is_gzipped_default)
                loaded_in_data = flex_load(file_path, default_serializer=serializer, default_is_gzipped=is_gzipped_default)

                det_serializer, det_is_gzipped, det_is_binary = \
                    determine_serialization(file_path, default_serializer=serializer, default_is_gzipped=is_gzipped_default)
                self.assertEqual(det_is_gzipped, is_actually_gzipped)
                errors = object_hierarchy_equals(loaded_in_data, self.test_data)
                print(yaml.dump(errors, default_flow_style = False))
                self.assertEqual(len(errors), 0)
                rm(file_path)

                # print("ERROR")
                # print("path", file_path)
                # print("seri", file.serializer)
                # print("bin ", file.is_binary)
                # print("gzip", file.is_gzipped)
                # print("ig  ", is_gzipped)
                # print("igd ", is_gzipped_default)
                # a = sys.exc_info()
                # print("".join(format_exception(*a)))
                # print("")
                # print("")




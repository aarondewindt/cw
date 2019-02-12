import unittest
from io import BytesIO
from pathlib import Path

from cw.xsens import XSensParser, XDI, StatusWordType

from cw.object_hierarchies import object_hierarchy_equals
from cw.flex_file import flex_load

import pprint

pprint = pprint.PrettyPrinter(indent=4).pprint

test_dir_path = Path(__file__).parent


class TestXSens(unittest.TestCase):
    def setUp(self):
        self.correct_single_packet = {
            XDI.packet_counter: 35216,
            XDI.sample_time: 254.1363,
            XDI.delta_v: (8.416175842285156e-05,
                          -0.00010259449481964111,
                          0.024537160992622375),
            XDI.acceleration: (0.03378037363290787,
                               -0.04108057916164398,
                               9.814864158630371),
            XDI.rate_ofturn: (-0.0017493965569883585,
                              -0.0047072782181203365,
                              -0.0016480687772855163),
            XDI.delta_q: (1.0000001192092896,
                          -2.186745632570819e-06,
                          -5.884096026420593e-06,
                          -2.0600855350494385e-06),
            XDI.status_word: (True, False, True, 1, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 0)}

    def test_parse_single_message(self):
        xp = XSensParser(None)

        data = bytes.fromhex("FA FF 36 53 10 20 02 89 90 10 60 04 00 26 C7 33 40 20 0C 3D 0A 5D 4A BD 28 44 1C 41 1D "
                             "09 AF 40 10 0C 38 B0 80 00 B8 D7 28 00 3C C9 02 28 80 20 0C BA E5 4C 02 BB 9A 3F 83 BA "
                             "D8 04 03 80 30 10 3F 80 00 01 B6 12 C0 01 B6 C5 70 00 B6 0A 40 00 E0 20 04 00 00 00 05 "
                             "F4")

        xp.parse(data)

        packet = xp.get()
        errors = object_hierarchy_equals(packet, self.correct_single_packet)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

    def test_parse_single_message_with_leading_bytes(self):
        xp = XSensParser(None)

        data = bytes.fromhex("B0 80 00 B8 D7 28 00 3C C9 02 28"
                             "FA FF 36 53 10 20 02 89 90 10 60 04 00 26 C7 33 40 20 0C 3D 0A 5D 4A BD 28 44 1C 41 1D "
                             "09 AF 40 10 0C 38 B0 80 00 B8 D7 28 00 3C C9 02 28 80 20 0C BA E5 4C 02 BB 9A 3F 83 BA "
                             "D8 04 03 80 30 10 3F 80 00 01 B6 12 C0 01 B6 C5 70 00 B6 0A 40 00 E0 20 04 00 00 00 05 "
                             "F4 ")

        xp.parse(data)

        packet = xp.get()
        errors = object_hierarchy_equals(packet, self.correct_single_packet)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

    def test_parse_single_message_trailing_bytes(self):
        xp = XSensParser(None)

        data = bytes.fromhex("FA FF 36 53 10 20 02 89 90 10 60 04 00 26 C7 33 40 20 0C 3D 0A 5D 4A BD 28 44 1C 41 1D "
                             "09 AF 40 10 0C 38 B0 80 00 B8 D7 28 00 3C C9 02 28 80 20 0C BA E5 4C 02 BB 9A 3F 83 BA "
                             "D8 04 03 80 30 10 3F 80 00 01 B6 12 C0 01 B6 C5 70 00 B6 0A 40 00 E0 20 04 00 00 00 05 "
                             "F4 02 89 90 10 60 04 00 26 C7 33 40 20 0C ")

        xp.parse(data)

        packet = xp.get()
        errors = object_hierarchy_equals(packet, self.correct_single_packet)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

    def test_parse_two_packets(self):
        xp = XSensParser(None)

        data = bytes.fromhex(
            "FA FF 36 53 10 20 02 89 90 10 60 04 00 26 C7 33 40 20 0C 3D 0A 5D 4A BD 28 44 1C 41 1D "
            "09 AF 40 10 0C 38 B0 80 00 B8 D7 28 00 3C C9 02 28 80 20 0C BA E5 4C 02 BB 9A 3F 83 BA "
            "D8 04 03 80 30 10 3F 80 00 01 B6 12 C0 01 B6 C5 70 00 B6 0A 40 00 E0 20 04 00 00 00 05 "
            "F4"
            "FA FF 36 53 10 20 02 89 90 10 60 04 00 26 C7 33 40 20 0C 3D 0A 5D 4A BD 28 44 1C 41 1D "
            "09 AF 40 10 0C 38 B0 80 00 B8 D7 28 00 3C C9 02 28 80 20 0C BA E5 4C 02 BB 9A 3F 83 BA "
            "D8 04 03 80 30 10 3F 80 00 01 B6 12 C0 01 B6 C5 70 00 B6 0A 40 00 E0 20 04 00 00 00 05 "
            "F4"
        )

        xp.parse(data)

        packet1 = xp.get()
        packet2 = xp.get()

        errors = object_hierarchy_equals(packet1, self.correct_single_packet)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

        errors = object_hierarchy_equals(packet2, self.correct_single_packet)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

    def test_single_packet_two_blocks(self):
        xp = XSensParser(None)

        data1 = bytes.fromhex("FA FF 36 53 10 20 02 89 90 10 60 04 00 26 C7 33 40 20 0C 3D 0A 5D 4A BD 28 44 1C 41 1D "
                              "09 AF 40 10 0C 38 B0 80 00 B8 D7 28 00 3C C9 02 28 80 20 0C BA E5 4C 02 BB 9A 3F 83 BA "
                              )

        data2 = bytes.fromhex("D8 04 03 80 30 10 3F 80 00 01 B6 12 C0 01 B6 C5 70 00 B6 0A 40 00 E0 20 04 00 00 00 05 "
                              "F4")

        xp.parse(data1)
        xp.parse(data2)

        packet = xp.get()
        errors = object_hierarchy_equals(packet, self.correct_single_packet)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

    def test_single_packet_file_like(self):
        bytes_io = BytesIO(bytes.fromhex(
            "FA FF 36 53 10 20 02 89 90 10 60 04 00 26 C7 33 40 20 0C 3D 0A 5D 4A BD 28 44 1C 41 1D "
            "09 AF 40 10 0C 38 B0 80 00 B8 D7 28 00 3C C9 02 28 80 20 0C BA E5 4C 02 BB 9A 3F 83 BA "
            "D8 04 03 80 30 10 3F 80 00 01 B6 12 C0 01 B6 C5 70 00 B6 0A 40 00 E0 20 04 00 00 00 05 "
            "F4")
        )

        xp = XSensParser(bytes_io)
        xp.start()

        packet = xp.get()
        errors = object_hierarchy_equals(packet, self.correct_single_packet)
        # print("\n".join(errors))
        self.assertEqual(len(errors), 0)

    def test_file(self):
        with open(test_dir_path / "data_files" / "xsens_raw_data.bin", "rb") as f:
            xp = XSensParser(f)
            xp.start()

            tables = xp.get_tables()

            # Uncomment this line to update the tables in the data file.
            # File(test_dir_path / "data_files" / "correct_xsens_tables.msgp.gz").dump(tables)

            correct_tables = flex_load(test_dir_path / "data_files" / "correct_xsens_tables.msgp.gz")

            errors = object_hierarchy_equals(tables, correct_tables)
            # print("\n".join(errors))
            self.assertEqual(len(errors), 0)

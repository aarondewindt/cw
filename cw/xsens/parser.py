import numpy as np
from io import BytesIO
from enum import Enum
from typing import Union, DefaultDict, Dict, Sequence
import struct
from collections import namedtuple, deque, defaultdict
from functools import partial
import tqdm

state_idle = 0
state_reading_header = 1
state_reading_data = 2
state_skipping_msg = 3


class XDI(Enum):
    """
    Xsens data identifier enumerator.
    """
    none = 0x0000
    type_mask = 0xfe00
    full_type_mask = 0xfff0
    full_mask = 0xffff
    format_mask = 0x01ff
    data_format_mask = 0x000f
    sub_format_mask = 0x0003
    sub_format_float = 0x0000
    sub_format_fp1220 = 0x0001
    sub_format_fp1632 = 0x0002
    sub_format_double = 0x0003
    temperature_group = 0x0800
    temperature = 0x0810
    timestamp_group = 0x1000
    utc_time = 0x1010
    packet_counter = 0x1020
    itow = 0x1030
    gps_age = 0x1040
    gnss_age = 0x1040
    pressure_age = 0x1050
    sample_time_fine = 0x1060
    sample_time_coarse = 0x1070
    sample_time = 0x10F0
    frame_range = 0x1080
    packet_counter8 = 0x1090
    sample_time64 = 0x10a0
    orientation_group = 0x2000
    coord_sys_mask = 0x000c
    coord_sys_enu = 0x0000
    coord_sys_ned = 0x0004
    coord_sys_nwu = 0x0008
    quaternion = 0x2010
    rotation_matrix = 0x2020
    euler_angles = 0x2030
    pressure_group = 0x3000
    baro_pressure = 0x3010
    acceleration_group = 0x4000
    delta_v = 0x4010
    acceleration = 0x4020
    free_acceleration = 0x4030
    acceleration_hr = 0x4040
    position_group = 0x5000
    altitude_msl = 0x5010
    altitude_ellipsoid = 0x5020
    position_ecef = 0x5030
    lat_lon = 0x5040
    snapshot_group = 0xc800
    retransmission_mask = 0x0001
    retransmission_flag = 0x0001
    awinda_snapshot = 0xc810
    full_snapshot = 0xc820
    gnss_group = 0x7000
    gnss_pvt_data = 0x7010
    gnss_sat_info = 0x7020
    angular_velocity_group = 0x8000
    rate_ofturn = 0x8020
    delta_q = 0x8030
    rate_ofturn_hr = 0x8040
    gps_group = 0x8800
    gps_dop = 0x8830
    gps_sol = 0x8840
    gps_time_utc = 0x8880
    gps_svinfo = 0x88a0
    raw_sensor_group = 0xa000
    raw_unsigned = 0x0000
    raw_signed = 0x0001
    raw_acc_gyr_mag_temp = 0xa010
    raw_gyro_temp = 0xa020
    raw_acc = 0xa030
    raw_gyr = 0xa040
    raw_mag = 0xa050
    raw_delta_q = 0xa060
    raw_delta_v = 0xa070
    raw_blob = 0xa080
    analog_ingroup = 0xb000
    analog_in1 = 0xb010
    analog_in2 = 0xb020
    magnetic_group = 0xc000
    magnetic_field = 0xc020
    velocity_group = 0xd000
    velocity_xyz = 0xd010
    status_group = 0xe000
    status_byte = 0xe010
    status_word = 0xe020
    rssi = 0xe040
    device_id = 0xe080
    indication_group = 0x4800
    trigger_in1 = 0x4810
    trigger_in2 = 0x4820

StatusWordType = namedtuple("StatusWordType", (
    "self_test",
    "filter_valid",
    "gnss_fix",
    "no_rotation_update_status",
    "timestamp_gnss_synced",
    "timestamp_clock_synced",
    "on_off",
    "clip_acc_x",
    "clip_acc_y",
    "clip_acc_z",
    "clip_gyr_x",
    "clip_gyr_y",
    "clip_gyr_z",
    "clip_mag_x",
    "clip_mag_y",
    "clip_mag_z",
    "clipping_indication",
    "sync_in_marker",
    "sync_out_marker",
    "filter_mode",
))


def decode_status_word(format, data):
    status_word_int = struct.unpack(">I", data)[0]

    # Keeping this comment here so I can remember which one is which until I write the documentation.
    # return StatusWordType(
    #     self_test=bool(status_word_int & 1),
    #     filter_valid=bool(status_word_int & (1 << 1)),
    #     gnss_fix=bool(status_word_int & (1 << 2)),
    #     no_rotation_update_status=(status_word_int & (0b11 << 2)) >> 2,
    #     timestamp_gnss_synced=bool(status_word_int & (1 << 5)),
    #     timestamp_clock_synced=bool(status_word_int & (1 << 6)),
    #     on_off=bool(status_word_int & (1 << 7)),
    #     clip_acc_x=bool(status_word_int & (1 << 8)),
    #     clip_acc_y=bool(status_word_int & (1 << 9)),
    #     clip_acc_z=bool(status_word_int & (1 << 10)),
    #     clip_gyr_x=bool(status_word_int & (1 << 11)),
    #     clip_gyr_y=bool(status_word_int & (1 << 12)),
    #     clip_gyr_z=bool(status_word_int & (1 << 13)),
    #     clip_mag_x=bool(status_word_int & (1 << 14)),
    #     clip_mag_y=bool(status_word_int & (1 << 15)),
    #     clip_mag_z=bool(status_word_int & (1 << 16)),
    #     clipping_indication=bool(status_word_int & (1 << 19)),
    #     sync_in_marker=bool(status_word_int & (1 << 21)),
    #     sync_out_marker=bool(status_word_int & (1 << 22)),
    #     filter_mode=(status_word_int & (0b11 << 22)) >> 22,
    # )

    return [
        bool(status_word_int & 1),
        bool(status_word_int & (1 << 1)),
        bool(status_word_int & (1 << 2)),
        (status_word_int & (0b11 << 2)) >> 2,
        bool(status_word_int & (1 << 5)),
        bool(status_word_int & (1 << 6)),
        bool(status_word_int & (1 << 7)),
        bool(status_word_int & (1 << 8)),
        bool(status_word_int & (1 << 9)),
        bool(status_word_int & (1 << 10)),
        bool(status_word_int & (1 << 11)),
        bool(status_word_int & (1 << 12)),
        bool(status_word_int & (1 << 13)),
        bool(status_word_int & (1 << 14)),
        bool(status_word_int & (1 << 15)),
        bool(status_word_int & (1 << 16)),
        bool(status_word_int & (1 << 19)),
        bool(status_word_int & (1 << 21)),
        bool(status_word_int & (1 << 22)),
        (status_word_int & (0b11 << 22)) >> 22,
    ]


def decode_real(format, data, length):
    format_string = ">"
    if format == XDI.sub_format_float.value:
        format_string += "f" * length
    elif format == XDI.sub_format_double.value:
        format_string += "d" * length
    else:
        return np.nan

    result = struct.unpack(format_string, data)
    if length == 1:
        return result[0]
    else:
        return result

mtdata2_decode = {
    # XDI.none: None,
    # XDI.type_mask: None,
    # XDI.full_type_mask: None,
    # XDI.full_mask: None,
    # XDI.format_mask: None,
    #
    # XDI.data_format_mask: None,
    # XDI.sub_format_mask: None,
    # XDI.sub_format_float: None,
    # XDI.sub_format_fp1220: None,
    # XDI.sub_format_fp1632: None,
    # XDI.sub_format_double: None,
    #
    # XDI.temperature_group: None,
    XDI.temperature: lambda format, data: decode_real(format, data, 1),

    # XDI.timestamp_group: None,
    XDI.utc_time: lambda format, data: struct.unpack(">IHBBBBBB", data),
    XDI.packet_counter: lambda format, data: struct.unpack(">H", data)[0],
    XDI.itow: lambda format, data: struct.unpack(">I", data)[0],
    XDI.gps_age: lambda format, data: struct.unpack(">B", data)[0],
    XDI.gnss_age: lambda format, data: struct.unpack(">H", data)[0],
    XDI.pressure_age: lambda format, data: struct.unpack(">H", data)[0],
    XDI.sample_time_fine: lambda format, data: struct.unpack(">I", data)[0],
    XDI.sample_time_coarse: lambda format, data: struct.unpack(">I", data)[0],
    XDI.frame_range: lambda format, data: struct.unpack(">HH", data),
    XDI.packet_counter8: lambda format, data: struct.unpack(">B", data)[0],
    XDI.sample_time64: lambda format, data: struct.unpack(">Q", data)[0],

    # XDI.orientation_group: None,
    XDI.coord_sys_mask: None,
    XDI.coord_sys_enu: None,
    XDI.coord_sys_ned: None,
    XDI.coord_sys_nwu: None,
    XDI.quaternion: None,
    XDI.rotation_matrix: None,
    XDI.euler_angles: None,

    # XDI.pressure_group: None,
    XDI.baro_pressure: lambda format, data: struct.unpack(">I", data)[0],

    # XDI.acceleration_group: None,
    XDI.delta_v: lambda format, data: decode_real(format, data, 3),
    XDI.acceleration: lambda format, data: decode_real(format, data, 3),
    XDI.free_acceleration: lambda format, data: decode_real(format, data, 3),
    # XDI.acceleration_hr: None,

    # XDI.position_group: None,
    # XDI.altitude_msl: ,
    # XDI.altitude_ellipsoid: ,
    # XDI.position_ecef: ,
    # XDI.lat_lon: ,

    # XDI.snapshot_group: ,
    # XDI.retransmission_mask: ,
    # XDI.retransmission_flag: ,
    # XDI.awinda_snapshot: ,
    # XDI.full_snapshot: ,

    # XDI.gnss_group: None,
    # XDI.gnss_pvt_data: ,
    # XDI.gnss_sat_info: ,

    # XDI.angular_velocity_group: None,
    XDI.rate_ofturn: lambda format, data: decode_real(format, data, 3),
    XDI.delta_q: lambda format, data: decode_real(format, data, 4),
    # XDI.rate_ofturn_hr: ,

    # XDI.gps_group: None,
    # XDI.gps_dop: ,
    # XDI.gps_sol: ,
    # XDI.gps_time_utc: ,
    # XDI.gps_svinfo: ,

    # XDI.raw_sensor_group: None,
    # XDI.raw_unsigned: ,
    # XDI.raw_signed: ,
    XDI.raw_acc_gyr_mag_temp: lambda format, data: struct.unpack(">HHHHHHHHHH", data),
    XDI.raw_gyro_temp: lambda format, data: struct.unpack(">HHH", data),
    XDI.raw_acc: lambda format, data: struct.unpack(">HHH", data),
    XDI.raw_gyr: lambda format, data: struct.unpack(">HHH", data),
    XDI.raw_mag: lambda format, data: struct.unpack(">HHH", data),
    # XDI.raw_delta_q: ,
    # XDI.raw_delta_v: ,
    # XDI.raw_blob: ,

    # XDI.analog_ingroup: None,
    # XDI.analog_in1: ,
    # XDI.analog_in2: ,

    # XDI.magnetic_group: None,
    XDI.magnetic_field: lambda format, data: decode_real(format, data, 3),

    # XDI.velocity_group: None,
    XDI.velocity_xyz: lambda format, data: decode_real(format, data, 3),

    # XDI.status_group: None,
    # XDI.status_byte: None,
    XDI.status_word: decode_status_word,
    # XDI.rssi: None,
    # XDI.device_id: None,
    #
    # XDI.indication_group: None,
    # XDI.trigger_in1: None,
    # XDI.trigger_in2: None,
}


class XSensParser:
    """
    Class that parses Xsens messages from an incoming stream of bytes.

    :param file_like: Binary file like object with the xsens data.
    :param max_size: Maximum size of the deque holding the packages found.
    :param calibration_data: Dictionary holding the xsens calibration data.
       This parameter is only required if the raw sensor data was recorded.
    :param verbose: If True, a progressbar will be shown.

    """

    def __init__(self, file_like, max_size: int = None, calibration_data: dict =None, verbose=False):
        self.byte_buffer = BytesIO()
        self.state: int = state_idle
        self.bus_id: int = None
        self.msg_id: int = None
        self.data_length: int = None
        self.bytes_left_to_read: int = None
        self.mtdata2_deque = deque(maxlen=max_size)
        self.file_like = file_like
        self.calibration_data = calibration_data
        self.device_id: Union[None, str] = None
        self.verbose = verbose

        if calibration_data is not None:
            # Convert calibration data to numpy arrays and calculate gain matrix
            for top_level_key in ['accelerometer', 'rate_gyro', 'magnetometer']:
                for key, value in calibration_data[top_level_key].items():
                    self.calibration_data[top_level_key][key] = \
                        np.array(calibration_data[top_level_key][key])
                self.calibration_data[top_level_key]['k_t_inv'] = \
                    np.linalg.inv(
                        np.diag(self.calibration_data[top_level_key]['gains']) @ \
                        self.calibration_data[top_level_key]['alignment_matrix']
                    )

    def __len__(self):
        return len(self.mtdata2_deque)

    def get(self):
        return self.mtdata2_deque.popleft()

    def start(self):
        total_size = None
        try:
            self.file_like.seek(0, 2)
            total_size = self.file_like.tell()
            self.file_like.seek(0)
        except:
            pass

        with tqdm.tqdm(total=total_size, desc="parsing", disable=(not self.verbose), unit="bytes") as pbar:
            for chunk in iter(partial(self.file_like.read, 1024), b''):
                pbar.update(len(chunk))
                self.parse(chunk)

    @property
    def count(self):
        return len(self.mtdata2_deque)

    def reset(self):
        self.state = state_idle
        self.bus_id = None
        self.msg_id = None
        self.data_length = None
        self.bytes_left_to_read = None
        self.clear_buffer()

    def clear_buffer(self):
        self.byte_buffer.seek(0)
        self.byte_buffer.truncate(0)

    def parse(self, bts: Union[bytes, bytearray]):
        i = 0

        while i < len(bts):
            if self.state == state_idle:
                # In idle, so look for preamble byte 0xFA
                if 0xFA != bts[i]:
                    # If the byte is not the preamble byte. Skip it.
                    pass
                else:
                    # We are now going to head the header
                    # Set the current reading index to the first byte in the header.
                    # The header is 3 bytes long
                    # Clear the byte buffer
                    self.state = state_reading_header
                    self.bytes_left_to_read = 3
                    self.clear_buffer()
                i += 1

            elif self.state == state_reading_header:

                # Get the header bytes from bts
                i_end = len(bts) if (i + self.bytes_left_to_read) >= len(bts) else (i + self.bytes_left_to_read)
                header_bytes = bts[i: i_end]
                n_bytes_read = self.byte_buffer.write(header_bytes)
                i += n_bytes_read
                self.bytes_left_to_read -= n_bytes_read

                if self.bytes_left_to_read <= 0:
                    header_bytes = self.byte_buffer.getvalue()
                    # print(header_bytes)
                    self.bus_id, self.msg_id, self.data_length = header_bytes
                    self.clear_buffer()

                    # If it's a MTData2 messages, read it, otherwise skip it.
                    if self.msg_id in self.message_handlers:
                        self.state = state_reading_data
                    else:
                        self.state = state_skipping_msg

                    # Read/skip the data block and the checksum byte.
                    self.bytes_left_to_read = self.data_length + 1

            elif self.state == state_skipping_msg:
                n_skipping = len(bts) if self.bytes_left_to_read > len(bts) else self.bytes_left_to_read
                i += n_skipping
                self.bytes_left_to_read -= n_skipping

                # If there are no more bytes to skip, reset the parser and go back to the idle state.
                if self.bytes_left_to_read <= 0:
                    self.reset()

            elif self.state == state_reading_data:
                i_end = len(bts) if (i + self.bytes_left_to_read) >= len(bts) else (i + self.bytes_left_to_read)
                data_bytes = bts[i: i_end]
                n_bytes_read = self.byte_buffer.write(data_bytes)
                i += n_bytes_read
                self.bytes_left_to_read -= n_bytes_read

                if self.bytes_left_to_read <= 0:
                    # Get the data and checksum from the buffer
                    data = self.byte_buffer.getvalue()
                    data_field, checksum = data[:-1], data[-1]

                    # Check the integrity of the package.
                    # This is done by doing an 8bit unsigned integer summation of
                    # all bytes, except for the preamble. the sum must be zero.
                    sum = (((self.bus_id + self.msg_id) % 256 + self.data_length) % 256 + checksum) % 256
                    for byte in data_field:
                        sum = (sum + byte) % 256

                    if sum == 0:
                        self.message_handlers[self.msg_id](self, data_field)

                    self.reset()

    def handle_mtdata2(self, data_field):
        """
        Parses MTdata2 packages. The package is in the end appended to mtdata2_deque.

        :param data_field: Byte string with the message data field.
        """
        i = 0
        len_data_field = len(data_field)
        package = {}

        while i < len_data_field:
            data_id_field = struct.unpack(">H", data_field[i:i+2])[0]
            data_id = XDI(data_id_field & XDI.full_type_mask.value)

            data_format = data_id_field & XDI.data_format_mask.value
            data_length = data_field[i+2]

            if data_id in mtdata2_decode:
                package[data_id] = mtdata2_decode[data_id](data_format, data_field[i+3:i+3+data_length])

            i += 3 + data_length

        # If raw sensor data was given, then is has to be calibrated using the calibration data.
        if XDI.raw_acc_gyr_mag_temp in package:
            if self.calibration_data is None:
                raise Exception("Calibration data required, raw sensor data was logged.")

        # Get sample time. If both the fine and coarse sample time is given,
        # calculate the "big time". If only one of them is given. Use it.
        if (XDI.sample_time_fine in package) and (XDI.sample_time_coarse in package):
            time = package[XDI.sample_time_coarse] + ((package[XDI.sample_time_fine] % 10000) / 10000)
            del package[XDI.sample_time_fine]
            del package[XDI.sample_time_coarse]
        elif XDI.sample_time_fine in package:
            time = package[XDI.sample_time_fine] / 10000
            del package[XDI.sample_time_fine]
        elif XDI.sample_time_coarse in package:
            time = float(package[XDI.sample_time_coarse])
            del package[XDI.sample_time_coarse]
        else:
            raise Exception("No sample time information.")

        package[XDI.sample_time] = time

        def calibrate_data(data: Sequence, calibration_data):
            return calibration_data['k_t_inv'] @ (data - calibration_data['offsets'])

        if XDI.raw_acc_gyr_mag_temp in package:
            raw_imu = package[XDI.raw_acc_gyr_mag_temp]
            package[XDI.acceleration] = calibrate_data(
                raw_imu[0:3],
                self.calibration_data['accelerometer']
            ).tolist()
            package[XDI.rate_ofturn] = calibrate_data(
                raw_imu[3:6],
                self.calibration_data['rate_gyro']
            ).tolist()
            package[XDI.magnetic_field] = calibrate_data(
                raw_imu[6:9],
                self.calibration_data['magnetometer']
            ).tolist()
            package[XDI.temperature] = raw_imu[9] * 0.00390625  # Convert to celsius
            del package[XDI.raw_acc_gyr_mag_temp]

        self.mtdata2_deque.append(package)

    def handle_device_id(self, data_field):
        self.device_id = data_field.hex().upper()
        if self.calibration_data:
            if self.device_id != self.calibration_data['device_id']:
                raise Exception("Logged data device id does not match the calibration data device id.")

    def get_tables(self):
        """
        Generates and returns a dictionary containing all the of the data in mtdata2_deque.

        :return: Dictionary with the parsed data.
        """

        # Create a default dictionary that will be used to create the tables.
        tables: DefaultDict[XDI, Dict[str, Sequence]] = defaultdict(lambda: {"time": deque(), "data": deque()})

        # These data field should be excluded from the final table.
        exclude_data_ids = [XDI.sample_time]  # []

        # If raw sensor data was given, then is has to be calibrated using the calibration data.
        # Check if the calibration data was given.
        # Check whether the device_id in the calibration data matches the device id in the logged data.
        if XDI.raw_acc_gyr_mag_temp in self.mtdata2_deque[-1]:
            if self.calibration_data is None:
                raise Exception("Calibration data required, raw sensor data was logged.")
            else:
                if self.device_id is not None:
                    if self.device_id != self.calibration_data['device_id']:
                        raise Exception("The device id of the logged data does not match that of the calibration data.")
                else:
                    print("WARNING: Logged data does not contain device_id.")

        with tqdm.tqdm(total=len(self.mtdata2_deque), desc="creating tables", disable=(not self.verbose), unit="pkts") as pbar:
            for packet in self.mtdata2_deque:
                pbar.update(1)

                # Loop through all packages.
                for data_id, value in packet.items():
                    # Skip excluded data fields/
                    if data_id in exclude_data_ids:
                        continue

                    tables[data_id]['time'].append(packet[XDI.sample_time])
                    tables[data_id]['data'].append(value)

        # The data type of 'tables' and it's content where chosen for performance and ease
        return_dict = {"device_id": self.device_id}
        for data_id, value in tables.items():
            return_dict[data_id.name] = {"time": list(value['time']), "data": list(value['data'])}

        return return_dict

    message_handlers = {
        0x36: handle_mtdata2,
        0x01: handle_device_id
    }

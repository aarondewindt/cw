from io import BytesIO
from enum import Enum
import serial_asyncio
import asyncio


class RxState(Enum):
    idle = 0
    receiving = 1
    escaping = 2
    complete = 3


class LBPAsyncDevice:
    def __init__(self, url, baudrate=38400):
        self.url = url
        self.baudrate = baudrate
        self.reader = None
        self.writer = None
        self.package_queue = None

    async def start(self):
        self.reader, self.writer = await serial_asyncio.open_serial_connection(url=self.url, baudrate=self.baudrate)
        self.package_queue = asyncio.Queue()
        asyncio.create_task(self.reader_coro())

    async def reader_coro(self):
        lbp = LBPPacket()
        while True:
            byte = (await self.reader.readexactly(1))[0]
            done = lbp.parse_byte(byte)
            if done:
                await self.package_queue.put(lbp)
                lbp = LBPPacket()

    async def transmit(self,
                       command,
                       destination,
                       source,
                       flags,
                       sequence,
                       data):
        lbp = LBPPacket(command, destination, source, flags, sequence, data)
        self.writer.write(lbp.serialize())

    async def transmit_package(self, package):
        self.writer.write(package.serialize())


class LBPPacket:
    def __init__(self,
                 command=None,
                 destination=None,
                 source=None,
                 flags=None,
                 sequence=None,
                 data=None):
        self.buffer = BytesIO()
        self._rx_crc = 0
        self._rx_state: RxState = RxState.idle

        self.header = bytearray([0, 0, 0])
        self.data = None

        self.command = command or 0
        self.destination = destination or 0x3F
        self.source = source or 0x3F
        self.flags = flags or 0x00
        self.sequence = sequence or 0
        self.data = data or bytearray()

    def parse_byte(self, b):
        if self._rx_state is RxState.complete:
            return True

        if b == Comms.PACKET_START:
            self.buffer.seek(0)
            self.buffer.truncate(0)
            self._rx_crc = 0
            self._rx_state = RxState.receiving
            return False

        if self._rx_state is RxState.idle:
            return False

        if b == Comms.PACKET_END:
            if not self._rx_crc:
                raw_data = self.buffer.getbuffer()
                self.header = bytearray(raw_data[:3])
                self.data = bytearray(raw_data[3:-1])
                del raw_data
                self.buffer.seek(0)
                self.buffer.truncate(0)
                self._rx_state = RxState.complete
                return True
            self._rx_state = RxState.idle
            return False

        if b == Comms.PACKET_ESCAPE:
            self._rx_state = RxState.escaping
            return False

        if self._rx_state is RxState.escaping:
            self._rx_state = RxState.receiving
            b = (~b) & 0xFF

        self.buffer.write(bytes([b]))
        self._rx_crc = crc8(b, self._rx_crc)
        return False

    def serialize(self):
        tx_buffer = bytearray((Comms.PACKET_START,))
        tx_crc = 0

        def add_byte(b):
            if (b == Comms.PACKET_START) or (b == Comms.PACKET_END) or (b == Comms.PACKET_ESCAPE):
                tx_buffer.append(Comms.PACKET_ESCAPE)
                tx_buffer.append((~b) & 0xFF)
            else:
                tx_buffer.append(b)
            return crc8(b, tx_crc)

        for byte in self.header:
            tx_crc = add_byte(byte)

        for byte in self.data:
            tx_crc = add_byte(byte)

        add_byte(tx_crc)
        tx_buffer.append(Comms.PACKET_END)

        return bytes(tx_buffer)

    def __str__(self):
        return "Launch Box Protocol Package\n"\
               f"  Flag: {Comms.name_from_code(self.flags)} ({hex(self.flags)})\n"\
               f"  Source: {hex(self.source)}\n"\
               f"  Sequence: {hex(self.sequence)}\n"\
               f"  Destination: {hex(self.destination)}\n"\
               f"  Command: {Comms.name_from_code(self.command)} ({hex(self.command)})\n"\
               f"  Data: {self.data}\n"

    @property
    def command(self):
        return self.header[2]

    @command.setter
    def command(self, value):
        self.header[2] = value

    @property
    def destination(self):
        return self.header[1] & Comms.ADDRESS_MASK

    @destination.setter
    def destination(self, value):
        self.header[1] = (self.header[1] & Comms.FLAGS_MASK) | (value & Comms.ADDRESS_MASK)

    @property
    def source(self):
        return self.header[0] & Comms.ADDRESS_MASK

    @source.setter
    def source(self, value):
        self.header[0] = (self.header[0] & Comms.FLAGS_MASK) | (value & Comms.ADDRESS_MASK)

    @property
    def flags(self):
        return self.header[0] & Comms.FLAGS_MASK

    @flags.setter
    def flags(self, value):
        self.header[0] = (value & Comms.FLAGS_MASK) | (self.header[0] & Comms.ADDRESS_MASK)

    @property
    def sequence(self):
        return (self.header[1] & Comms.FLAGS_MASK) >> 6

    @sequence.setter
    def sequence(self, value):
        self.header[1] = ((value << 6) & Comms.FLAGS_MASK) | (self.header[1] & Comms.ADDRESS_MASK)


class Comms(object):
    @classmethod
    def name_from_code(cls, code):
        for key, value in cls.__dict__.items():
            if value == code:
                return key
        return hex(code)

    ADDRESS_MASK = 0x3F
    FLAGS_MASK = 0xC0

    ADDRESS_UNKNOWN = 0x3F
    ADDRESS_SUP = 0x20
    ADDRESS_GROUND = 0x24
    ADDRESS_SERVER = 0x30
    ADDRESS_I2CSENSORS = 0x31
    ADDRESS_DATA_PROCESSOR = 0x32

    FLAGS_COMMAND = 0x00
    FLAGS_REPLY = 0x40
    FLAGS_NOTIFICATION = 0x80
    FLAGS_BROADCAST = 0xC0

    COMMAND_RESEND = 0x00
    COMMAND_NACK = 0x01
    COMMAND_IDENTIFY = 0x02
    COMMAND_IDENTIFY_REPLY = 0x03
    COMMAND_NETWORK_DISCOVERY = 0x04
    COMMAND_NETWORK_DISCOVERY_REPLY = 0x05
    COMMAND_STATUS = 0x06
    COMMAND_STATUS_REPLY = 0x07
    COMMAND_WINDOW_SIZE = 0x08

    COMMAND_PAUSE_TICK = 0x10
    COMMAND_RESUME_TICK = 0x11
    COMMAND_DATA_PROCESSOR_PACKET = 0x12

    COMMAND_READ_REGISTER = 0x71

    PACKET_START = 0x55
    PACKET_END = 0x5A
    PACKET_ESCAPE = 0x50

    DEVICE_STATUS_STANDARD = 0x00
    DEVICE_STATUS_ROCKET = 0x10

    DEVICE_STATUS_SAFE = 0x00
    DEVICE_STATUS_ARMED = 0x01

    DEVICE_STATUS_OK = 0x00
    DEVICE_STATUS_WARNING = 0x02
    DEVICE_STATUS_ERROR = 0x04


def crc8(b, crc):
    b2 = b
    if (b < 0):
        b2 = b + 256
    for i in range(8):
        odd = ((b2 ^ crc) & 1) == 1
        crc >>= 1
        b2 >>= 1
        if (odd):
            crc ^= 0x8C
    return crc


if __name__ == '__main__':
    lbp = LBPPacket(source=0x10,
                    destination=0x24,
                    flags=Comms.FLAGS_COMMAND,
                    sequence=0x00,
                    command=Comms.COMMAND_IDENTIFY,
                    data=b"")
    done = False

    # for i, byte in enumerate(b'\x55\x62\x24\x07\x00\x53\x74\x61\x74\x65\x5f\x30\x07\x5a'):
    #     done = lbp.parse_byte(byte)
    #     if done:
    #         break

    print(lbp)
    print(len(lbp.serialize()))

    for byte in lbp.serialize():
        print(byte, end=", ")
        # print(f"{byte:X}", end=" ")
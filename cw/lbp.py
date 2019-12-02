from io import BytesIO
from enum import Enum
from collections import defaultdict
from typing import DefaultDict, List, Union
import serial_asyncio
import asyncio


class RxState(Enum):
    idle = 0
    receiving = 1
    escaping = 2
    complete = 3


class LBPAsyncDevice:
    def __init__(self, url, baudrate=38400, assume_reply_equals_command=False, loop=None):
        self.url = url
        self.baudrate = baudrate
        self.reader = None
        self.writer = None
        self.loop = loop or asyncio.get_event_loop()
        self.assume_reply_equals_command = assume_reply_equals_command
        self.package_queue: asyncio.Queue = None
        self.sequence_queue: asyncio.Queue = None
        self.awaiting_replies: List[Union[asyncio.Future, None]] = [None, None, None, None]

    async def start(self):
        """
        Open the serial connection and listen for incoming packages.
        """
        self.reader, self.writer = await serial_asyncio.open_serial_connection(url=self.url, baudrate=self.baudrate)
        self.package_queue = asyncio.Queue()
        self.sequence_queue = asyncio.Queue()
        for i in range(4):
            self.sequence_queue.put_nowait(i)
        self.loop.create_task(self.reader_coro())

    def close(self):
        self.writer.transport.close()

    async def reader_coro(self):
        """
        Coroutine listening for packages. Replies will unblock their corresponding command
        transmit call and the rest of the packages will be places in the queue.
        """
        lbp = LBPPacket()
        while True:
            # Wait for one byte to arrive.
            byte = (await self.reader.readexactly(1))[0]

            # Process the byte.
            done = lbp.parse_byte(byte)

            # Process the package if a full package has been received.
            if done:
                # If this is a reply package, check if we are waiting for a reply from
                # this (command, sequence) combination. If so, set the future result
                # and continue the loop.
                if lbp.flags == Comms.FLAGS_REPLY:
                    if self.awaiting_replies[lbp.sequence] is not None:
                        future = self.awaiting_replies[lbp.sequence]
                        self.awaiting_replies[lbp.sequence] = None
                        try:
                            future.set_result(lbp)
                        except asyncio.InvalidStateError:
                            pass
                        lbp = LBPPacket()
                        continue

                # Put the package in the queue, if it wasn't an awaiting reply.
                await self.package_queue.put(lbp)

                # Initialize new a package.
                lbp = LBPPacket()

    async def transmit(self,
                       command=None,
                       destination=None,
                       source=None,
                       flags=None,
                       sequence=None,
                       data=None,
                       force_asynchronous=None):
        """
        Create and transmit an LBP package. If the package has a COMMAND flag, the package will be send
        synchronously. This means, the package sequence will be set, the package will be transmitted
        and this function will await until a reply with the same sequence number has been received.
        The coroutine will the return the reply package. If it's an asynchronous or reply package, the
        coroutine will await until the package has been transmitted and return None.

        :param command: Command code.
        :param destination: Destination address.
        :param source: Source address
        :param flags: Package flag.
        :param sequence: Package sequence.
        :param data: Package payload data.
        :param force_asynchronous: If True, command packages will be send asynchronously.
        :return: Reply package if the transmitted package was a command package, otherwise None.
        """
        # Create package
        package = LBPPacket(command, destination, source, flags, sequence, data)

        # Transmit, wait and return the reply if synchronous.
        return await self.transmit_package(package, force_asynchronous)

    async def transmit_package(self, package: 'LBPPacket', force_asynchronous: bool=False):
        """
        Transmits an LBP package. If the package has a COMMAND flag, the package will be send synchronously.
        This means, the package sequence will be set, the package will be transmitted and this function
        will await until a reply with the same sequence number has been received. The coroutine will the
        return the reply package. If it's an asynchronous or reply package, the coroutine will await until
        the package has been transmitted and return None.

        :param package: Package to transmit
        :param force_asynchronous: If True, command packages will be send asynchronously.

        :return: Reply package if the transmitted package was a command package, otherwise None.
        """
        is_synchronous = (package.flags == Comms.FLAGS_COMMAND) and (not force_asynchronous)

        if is_synchronous:
            # Wait for an available sequence for the command.
            sequence = await self.sequence_queue.get()

            # Set the package sequence.
            package.sequence = sequence

            # Create the future object used to wait for the reply.
            self.awaiting_replies[sequence] = asyncio.Future()

        # Transmit package.
        self.writer.write(package.serialize())

        if is_synchronous:
            # Wait for the reply.
            future = self.awaiting_replies[sequence]
            await future

            # Get the reply package.
            reply_package = future.result()

            # Clear the future and add the sequence back to the queue.
            self.awaiting_replies[sequence] = None
            await self.sequence_queue.put(sequence)

            # Return the reply.
            return reply_package


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


def lbp_listener(url, baudrate=38400, assume_reply_equals_command=False):
    """
    Listens and prints incoming lbp packages. Used

    :return:
    """

    async def listener():
        device = LBPAsyncDevice(url, baudrate, assume_reply_equals_command)
        await device.start()
        print(f"Listening for packages from '{url}' ({baudrate}).")

        while True:
            package = await device.package_queue.get()
            print(package)
            if package.flags == Comms.FLAGS_COMMAND:
                source = package.source
                package.source = package.destination
                package.destination = source
                package.flags = Comms.FLAGS_REPLY
                await device.transmit_package(package)

    asyncio.run(listener())


if __name__ == '__main__':
    # lbp_listener("/tmp/lbp_port_b")
    lbp = LBPPacket(source=0x24, destination=0x20)
    done = False

    byts = bytes([
        85,
        127,
        63,
        133,
        97,
        98,
        99,
        100,
        191,
        90,
    ])

    print(byts)

    print(byts[1])

    for i, byte in enumerate(byts): #b'U\x7f?\x85abcd\xbfZ'):
        done = lbp.parse_byte(byte)
        if done:
            break

    print(lbp)
    print(lbp.sequence)
    print(lbp.serialize())
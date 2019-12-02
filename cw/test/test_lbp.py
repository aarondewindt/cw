# Some tests require a serial connection between two virtual port. These can be started using
# the following command on a unix pc (linux, mac) with socat installed.
# socat -d -d -v pty,rawer,echo=0,link=/tmp/lbp_port_a pty,rawer,echo=0,link=/tmp/lbp_port_b

from typing import Tuple
import asyncio

import unittest

from cw.lbp import LBPPacket, LBPAsyncDevice, Comms
from cw.async_test import async_test


port_a_url = "/tmp/lbp_port_a"
port_b_url = "/tmp/lbp_port_b"


class TestLBP(unittest.TestCase):
    async def create_devices(self) -> Tuple[LBPAsyncDevice, LBPAsyncDevice]:
        device_a = LBPAsyncDevice(port_a_url)
        device_b = LBPAsyncDevice(port_b_url)

        await device_a.start()
        await device_b.start()

        return device_a, device_b

    @async_test()
    async def test_asynchronous_device_transfer(self):
        device_a, device_b = await self.create_devices()

        package = LBPPacket(command=0x33,
                            destination=0x30,
                            source=0x31,
                            flags=Comms.FLAGS_NOTIFICATION,
                            sequence=0x00,
                            data=b"aZynchronous")

        await device_a.transmit_package(package)
        received_package = await device_b.package_queue.get()

        self.assertEqual(package.command, received_package.command)
        self.assertEqual(package.destination, received_package.destination)
        self.assertEqual(package.source, received_package.source)
        self.assertEqual(package.flags, received_package.flags)
        self.assertEqual(package.sequence, received_package.sequence)
        self.assertEqual(package.data, received_package.data)

        device_a.close()
        device_b.close()

    @async_test()
    async def test_synchronous_device_transfer(self):
        device_a, device_b = await self.create_devices()

        package = LBPPacket(command=0x33,
                            destination=0x30,
                            source=0x31,
                            flags=Comms.FLAGS_COMMAND,
                            sequence=0x00,
                            data=b"Zynchronous")

        async def device_a_coroutine():
            reply = await device_a.transmit_package(package)
            print(reply)
            self.assertEqual(package.command, reply.command)
            self.assertEqual(package.destination, reply.destination)
            self.assertEqual(package.source, reply.source)
            self.assertEqual(Comms.FLAGS_REPLY, reply.flags)
            self.assertEqual(package.sequence, reply.sequence)
            self.assertEqual(package.data, reply.data)

        async def device_b_coroutine():
            received_package = await device_b.package_queue.get()
            received_package.flags = Comms.FLAGS_REPLY
            await device_b.transmit_package(received_package)

        await asyncio.gather(device_a_coroutine(), device_b_coroutine())

        device_a.close()
        device_b.close()
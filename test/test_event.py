import unittest
import asyncio
import threading

from cw.event import Event


class TestEvent(unittest.TestCase):
    def test_normal_callbacks(self):
        # Create mutable to pass to the handlers,  this variable will be used
        # to check whether the handlers where actually called.
        data = [None, None]

        # Create the handlers
        def handler_1(data):
            data[0] = 23

        def handler_2(data):
            data[1] = 78

        # Create an event
        event = Event()

        # Add the handlers tot he event
        event += handler_1, handler_2

        # Raise event.
        event(data)

        # Check whether the handlers where called.
        self.assertEquals(data, [23, 78])

    # This unit test ran successfully on 2017-01-15. The reason it's skipped is becuase
    # of the current setup with the event handler. The stop function does not seem to be threadsafe.
    # I'll have to think of a way stopping it.
    @unittest.skip
    def test_async_callbacks(self):
        class ThreadStuff:
            def __init__(self, semaphore: threading.Semaphore):
                self.loop = None
                self.semaphore = semaphore

        class AsyncioStuff():
            def __init__(self, semaphore_1: threading.Semaphore, semaphore_2: threading.Semaphore):
                self.semaphore_1 = semaphore_1
                self.semaphore_2 = semaphore_2
                self.reply_handler_1 = None
                self.reply_handler_2 = None

        def thread_target(thread_stuff):
            """
            Target for the thread running the asyncio event loop.

            :return:
            """

            thread_stuff.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(thread_stuff.loop)
            # print(3)
            thread_stuff.semaphore.release()
            # print(4)
            thread_stuff.loop.run_forever()
            print(999999)

        thread_stuff = ThreadStuff(threading.Semaphore(0))

        thread = threading.Thread(target=thread_target, args=(thread_stuff,))

        thread.start()

        assert thread_stuff.semaphore.acquire(timeout=1)
        assert thread_stuff.loop
        print("qwerty")
        async_stuff = AsyncioStuff(threading.Semaphore(0), threading.Semaphore(0))

        async def handler_1(async_stuff: AsyncioStuff):
            async_stuff.reply_handler_1 = "Hello"
            async_stuff.semaphore_1.release()

        async def handler_2(async_stuff: AsyncioStuff):
            async_stuff.reply_handler_2 = "World"
            async_stuff.semaphore_2.release()

        event = Event(thread_stuff.loop)

        event += handler_1, handler_2

        event(async_stuff)

        assert async_stuff.semaphore_1.acquire(timeout=1)
        assert async_stuff.semaphore_2.acquire(timeout=1)
        self.assertEquals(async_stuff.reply_handler_1, "Hello")
        self.assertEquals(async_stuff.reply_handler_2, "World")
        print("sdckomdsockmsd")
        thread_stuff.loop.stop()
        thread.join()




if __name__ == '__main__':
    unittest.main()

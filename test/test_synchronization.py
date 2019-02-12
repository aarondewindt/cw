import unittest
from threading import Thread
from time import sleep, time

from cw.synchronization import CheckInSemaphore


class TestSync(unittest.TestCase):
    def test_check_in_semaphore(self):
        def thread_target(chs: CheckInSemaphore, dt, token):
            chs.check_in(token)
            sleep(dt)
            chs.check_out(token)

        chs = CheckInSemaphore()
        t0 = time()

        thread = Thread(target=thread_target, args=(chs, 0.2, 1))
        thread.start()

        sleep(0.1)
        thread = Thread(target=thread_target, args=(chs, 0.2, 2))
        thread.start()

        thread = Thread(target=thread_target, args=(chs, 0.05, 3))
        thread.start()

        sleep(0.01)

        chs.wait()
        dt = time() - t0

        self.assertGreaterEqual(dt, .3)



if __name__ == '__main__':
    unittest.main()

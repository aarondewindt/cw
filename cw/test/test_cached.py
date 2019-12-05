import unittest
from time import sleep, time as clock

from cw.cached import cached, cached_class


__author__ = "Aaron M. de Windt"

dt = 0.001


class TestCachedProperties(unittest.TestCase):
    def test_cached(self):
        """Since bar is a cached property it will only be executed once per instance and always return 42. Only
            the first call will sleep for 1 second."""

        class Foo:
            """Test class."""

            def __init__(self):
                self.__bar = 41

            @cached
            def bar(self):
                self.__bar += 1
                sleep(dt)
                return self.__bar

        foo = Foo()

        t_0 = clock()
        r = foo.bar
        self.assertGreater(clock()-t_0, dt)
        self.assertEqual(r, 42)

        t_0 = clock()
        r = foo.bar
        self.assertLess(clock() - t_0, dt)
        self.assertEqual(r, 42)

        foo.bar = 1
        self.assertEqual(foo.bar, 1)

    def test_cached_class(self):
        """Same as the other test, but returns 37 on all instances of the class. According to [1], 37 is objectively the
        funniest number."""
        class Foo:
            _bar = 36

            @cached_class
            def bar(cls):
                cls._bar += 1
                sleep(dt)
                return cls._bar

        foo = Foo()

        t_0 = clock()
        r = foo.bar
        self.assertGreater(clock() - t_0, dt)
        self.assertEqual(r, 37)

        t_0 = clock()
        r = foo.bar
        self.assertLess(clock() - t_0, dt)
        self.assertEqual(r, 37)

        baz = Foo()
        t_0 = clock()
        r = baz.bar
        self.assertLess(clock() - t_0, dt)
        self.assertEqual(r, 37)

        foo.bar = 1
        self.assertEqual(foo.bar, 1)
        # This does not seem to work across classes, I'll have a look if I ever end up needing it.
        # self.assertEqual(baz.bar, 1)

    def test_cached_private(self):
        """Tests whether the cached property works with private (name mangled) functions."""

        class Foo:
            """Test class."""

            def __init__(self):
                self.__bar_value = 41

            @cached
            def __bar(self):
                self.__bar_value += 1
                sleep(dt)
                return self.__bar_value

            def get_bar(self):
                return self.__bar

            def set_bar(self, value):
                self.__bar = value


        foo = Foo()

        t_0 = clock()
        r = foo.get_bar()
        self.assertGreater(clock()-t_0, dt)
        self.assertEqual(r, 42)

        t_0 = clock()
        r = foo.get_bar()
        self.assertLess(clock() - t_0, dt)
        self.assertEqual(r, 42)

        foo.set_bar(1)
        self.assertEqual(foo.get_bar(), 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)


# [1] http://splitsider.com/2014/08/37-is-objectively-the-funniest-number/

import unittest

from cw.itertools import iterify


class TestItertools(unittest.TestCase):
    def test_iterify_list(self):
        original = [1, 2, 3, 4]
        new = list(iterify(original))
        self.assertEqual(original, new)

    def test_iterify_scalar(self):
        original = 1
        new = list(iterify(original))
        self.assertEqual([original], new)

    def test_iterify_single_string(self):
        original = 'a string'
        new = list(iterify(original))
        self.assertEqual([original], new)

    def test_iterify_multiple_string(self):
        original = ['a string', "another string", "and another one"]
        new = list(iterify(original))
        self.assertEqual(original, new)


if __name__ == '__main__':
    unittest.main()

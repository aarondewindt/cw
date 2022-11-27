import unittest


from cw.levenshtein import levenshtein_recursive, levenshtein_full_matrix, \
                           levenshtein_full_matrix_numba, levenshtein_two_rows, levenshtein_two_rows_numba


test_functions = (
    # ("recursive", levenshtein_recursive),
    ("levenshtein_full_matrix", levenshtein_full_matrix),
    ("levenshtein_full_matrix_numba", levenshtein_full_matrix_numba),
    ("levenshtein_two_rows", levenshtein_two_rows),
    ("levenshtein_two_rows_numba", levenshtein_two_rows_numba),
)

test_strings = (
    ("kitten", "sitting", 3),
    ("Saturday", "Sunday", 3),
    ("abcdef", "abccdef", 1),
    ("abcdef", "abdef", 1),
    ("abcdef", "abkdef", 1),
    ("abcdeg", "abkdef", 2),
    ("abcdeg22", "abkdef", 4),
    ("abcdeg", "abkdef22", 4),
    ("123abcdefg", "abfdefg123456", 10),
    ("adsckdsbkclehbkerhbvtrwbvitrwbviwerbvoiwerbvbsa;lkxmc liejnviwtrunbvgieubrvciebvierwubvebrg",
        "adsckdsbkflehbkerhbvtrwbvitrwbviwerbvoiwerbvbsa;lkxmc liejnviwtrunbvgieubrvciebvierwubvebra", 2),
    ("dlfknvldndfjbsbvhbdfjvhbdfjhvbjdfhvbjnkajnexowjqeoijfoeurhgiwuehgbiwvreoiquodiqecgoiqerglkv",
        "123dlfknvldndejbsbvhbdfjvhbdfjhvbjdfhvbjvkajnexowjqeoijfoeuqhgiwuehgbiwvresiquodiqecgotqergzkvwvh", 12)
)


class TestLevenshtein(unittest.TestCase):
    def test_levenshtein(self):
        for name, function in test_functions:
            with self.subTest(name=name):
                for a, b, delta in test_strings:
                    assert function(a, b) == delta
                    
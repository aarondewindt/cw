from collections.abc import Sequence

import numpy as np
import numba as nb


def levenshtein_recursive(a: Sequence, b: Sequence) -> int:
    """
    Computes the Levenshtein distance between two sequences
    using the recursive algorithm.
    """
    if len(a) == 0:
        return len(b)
    
    elif len(b) == 0:
        return len(a)

    elif a[0] == b[0]:
        return levenshtein_recursive(a[1:], b[1:])

    else:
        return 1 + min(
            levenshtein_recursive(a[1:], b),
            levenshtein_recursive(a, b[1:]),
            levenshtein_recursive(a[1:], b[1:]))


def levenshtein_full_matrix(a: Sequence, b: Sequence) -> int:
    """
    Computes the Levenshtein distance between two sequences
    using the full matrix (aka Wagner-Fischer) algorithm.
    """

    m = len(a)
    n = len(b)

    d = np.zeros((m + 1, n + 1), dtype=int)
    d[:, 0] = range(m + 1)
    d[0, :] = range(n + 1)

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            cost = (0 if a[i - 1] == b[j - 1] else 1)

            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + cost
            )

    return d[-1, -1]
    

@nb.jit(nopython=True, cache=True)
def levenshtein_full_matrix_numba(a: Sequence, b: Sequence) -> int:
    """
    Computes the Levenshtein distance between two sequences
    using the full matrix (aka Wagner-Fischer) algorithm.

    Optimized using numba.
    """
    m = len(a)
    n = len(b)

    d = np.zeros((m + 1, n + 1))
    for i in nb.prange(m + 1):
        d[i, 0] = i

    for j in nb.prange(n + 1):
        d[0, j] = j

    for j in nb.prange(1, n + 1):
        for i in nb.prange(1, m + 1):
            cost = (0 if a[i - 1] == b[j - 1] else 1)

            d[i, j] = min(
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + cost
            )

    return d[-1, -1]


def levenshtein_two_rows(a: Sequence, b: Sequence) -> int:
    """
    Computes the Levenshtein distance between two sequences
    using the two row matrix algorithm (modified Wagner-Fischer).
    """
    m = len(a)
    n = len(b)

    v1 = np.zeros((n + 1,), dtype=int)
    v0 = np.array(range(n + 1))

    for i in range(m):
        v1[0] = i + 1
        for j in range(n):
            deletion_cost = v0[j+1] + 1
            insertion_cost = v1[j] + 1
            substitution_cost = v0[j] + (0 if a[i] == b[j] else 1)

            v1[j+1] = min(deletion_cost, insertion_cost, substitution_cost)

        t = v0
        v0 = v1
        v1 = t

    return v0[-1]


@nb.jit(nopython=True, cache=True)
def levenshtein_two_rows_numba(a: Sequence, b: Sequence) -> int:
    """
    Computes the Levenshtein distance between two sequences
    using the two row matrix algorithm (modified Wagner-Fischer).

    Optimized using numba.
    """
    m = len(a)
    n = len(b)

    v1 = np.zeros((n + 1,))
    v0 = np.empty((n + 1,))
    for j in nb.prange(n + 1):
        v0[j] = j

    for i in nb.prange(m):
        v1[0] = i + 1
        for j in nb.prange(n):
            deletion_cost = v0[j+1] + 1
            insertion_cost = v1[j] + 1
            substitution_cost = v0[j] + (0 if a[i] == b[j] else 1)

            v1[j+1] = min(deletion_cost, insertion_cost, substitution_cost)

        t = v0
        v0 = v1
        v1 = t

    return v0[-1]


# Best performing implementations.
# The numba implementation only works for sequence types
# supported by numba
# The non-numba implementation works with any sequence type.
levenshtein = levenshtein_two_rows
levenshtein_numba = levenshtein_two_rows_numba


if __name__ == "__main__":
    from cw.context import time_it

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

    test_functions = (
        # ("recursive", levenshtein_recursive),
        ("levenshtein_full_matrix", levenshtein_full_matrix),
        ("levenshtein_full_matrix_numba", levenshtein_full_matrix_numba),
        ("levenshtein_two_rows", levenshtein_two_rows),
        ("levenshtein_two_rows_numba", levenshtein_two_rows_numba),
    )

    
    # print(levenshtein_full_matrix(test_strings[8][0], test_strings[8][1]))
    # print(levenshtein_two_rows_diag(test_strings[8][0], test_strings[8][1]))
    # print(levenshtein_two_rows_diag("sitting", "kitten"))

    n = 1000
    levenshtein_full_matrix_numba("a", "b")
    levenshtein_two_rows_numba("a", "b")
    # levenshtein_two_rows_diag_numba("a", "b")

    times = []
    for name, function in test_functions:
        with time_it(name):
            for i in range(n):
                for a, b, delta in test_strings:
                    assert function(a, b) == delta
                    

    # print(levenshtein_1(*test_strings[0]))
    # print(levenshtein_1(*test_strings[1]))


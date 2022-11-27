from collections.abc import Sequence

import numpy as np
import numba as nb


def levenshtein_1(a: Sequence, b: Sequence) -> int:
    """
    Computes the Levenshtein distance between two sequences.
    """
    if len(a) == 0:
        return len(b)
    
    elif len(b) == 0:
        return len(a)

    elif a[0] == b[0]:
        return levenshtein_1(a[1:], b[1:])

    else:
        return 1 + min(
            levenshtein_1(a[1:], b),
            levenshtein_1(a, b[1:]),
            levenshtein_1(a[1:], b[1:]))


def levenshtein_2(a: Sequence, b: Sequence) -> int:
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
def levenshtein_3(a: Sequence, b: Sequence) -> int:
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


def levenshtein_4(a: Sequence, b: Sequence) -> int:
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

        v0 = v1.copy()

    return v0[-1]


@nb.jit(nopython=True, cache=True)
def levenshtein_5(a: Sequence, b: Sequence) -> int:
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



if __name__ == "__main__":
    from cw.context import time_it

    test_strings = (
        ("kitten", "sitting", 3),
        ("Saturday", "Sunday", 3),
        ("abcdef", "abccdef", 1),
        ("abcdef", "abdef", 1),
        ("abcdef", "abkdef", 1),
        ("adsckdsbkclehbkerhbvtrwbvitrwbviwerbvoiwerbvbsa;lkxmc liejnviwtrunbvgieubrvciebvierwubvebrg",
         "adsckdsbkflehbkerhbvtrwbvitrwbviwerbvoiwerbvbsa;lkxmc liejnviwtrunbvgieubrvciebvierwubvebra", 2)
    )

    test_functions = (
        # ("recursive", levenshtein_1),
        ("iterative", levenshtein_2),
        ("ite_numba", levenshtein_3),
        ("ite_2_row", levenshtein_4),
        ("2_row_num", levenshtein_5)
    )

    
    # print(levenshtein_2("kitten", "sitting"))
    # print(levenshtein_4("kitten", "sitting"))

    n = 1000
    levenshtein_3("a", "b")
    levenshtein_5("a", "b")

    times = []
    for name, function in test_functions:
        with time_it(name):
            for i in range(n):
                for a, b, delta in test_strings:
                    assert function(a, b) == delta
                    

    # print(levenshtein_1(*test_strings[0]))
    # print(levenshtein_1(*test_strings[1]))


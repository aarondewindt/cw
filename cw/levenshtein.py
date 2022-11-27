from collections.abc import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """
    Computes the Levenshtein distance between two sequences.
    """
    if len(a) == 0:
        return len(b)
    
    elif len(b) == 0:
        return len(a)

    elif a[0] == b[0]:
        return levenshtein(a[1:], b[1:])

    else:
        return 1 + min(
            levenshtein(a[1:], b),
            levenshtein(a, b[1:]),
            levenshtein(a[1:], b[1:]))


import numpy as np


def levenshtein(a: Sequence, b: Sequence) -> int:
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





if __name__ == "__main__":
    print(levenshtein("kitten", "sitting"))

    print(levenshtein("Saturday", "Sunday"))


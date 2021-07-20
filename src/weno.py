from math import prod
from fractions import Fraction


def crj(k, j):
    return [
        sum(
            Fraction(
                sum(
                    prod(r - q + 1 for q in range(0, k + 1) if q not in (m, l))
                    for l in range(0, k + 1)
                    if l != m
                ),
                prod(m - l for l in range(0, k + 1) if l != m),
            )
            for m in range(j + 1, k + 1)
        )
        for r in range(-1, k)
    ]

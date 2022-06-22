from functools import cache
from itertools import product
from math import prod, factorial as fac
from fractions import Fraction


@cache
def c_j(k, j):
    """
    Given a cell at index i and the order of accuracy k, we choose a stencil based on
    r cells to the left and s cells to the right, and cell i itself if r, s >= 0.

      r + s + 1 = k.

    Given k cell averages,

      u_{i - r}, ... u_{i - r + k - 1}

    there are constants c_{rj} such that the reconstructed vale at the cell boundary x_{i + ½}

      u_{i + ½} = sum(c_{rj} * u_{i - r + j} for j in range(0, k))

    is k-th order accurate.

    Args:

        k: The order of accuracy, greater than or equal to one. Equal to the stencil width.
        j: The zero-based index within the stencil.

    Returns:
        A mapping from an integer left-shift of the stencil (r) to coefficients for the jth cell
        in the stencil.

    """
    if k < 1:
        raise ValueError(f"Order of accuracy (i.e. stencil width) k ({k}) is not positive")
    if not 0 <= j < k:
        raise ValueError(f"Stencil cell index j ({j}) out of range 0 <= j < k with k = {k}")
    return {
        r: sum(
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
    }


@cache
def c_rj(k, r, j):
    """"
    Args:
        k: The order of accuracy, greater than or equal to one. Equal to the stencil width.
        r: The left shift of the stencil in the range -1 <= r < k
        j: The zero-based index within the stencil in the range 0 <= j < k

    Returns:
        The coefficient for the cell average at position j with a stencil of order k shifted r
        cells to the right.
    """
    try:
        return c_j(k, j)[r]
    except KeyError:
        raise ValueError(f"r value ({r}) is not in the range -1 to k - 1 where k = {k}")


@cache
def c_r(k, r):
    """
    Args:
        k: The order of accuracy, greater than or equal to one. Equal to the stencil width.
        r: The left shift of the stencil in the range -1 <= r < k
    """
    return [c_rj(k, r, j) for j in range(0, k)]

def reconstruct_u_i_plus_half(us, i, k, r):
    """Reconstruct the state variable u at cell boundary i + 1/2.

    Args:
         us: A sequence of cell averages.
         i: The index of the cell for which the right boundary value is to be reconstructed.
         k: The order of accuracy, grater than equal to one. Equal to the stencil width.
         r: The right-shift in cells of the start of the stencil from i.
    """
    if not (0 <= r < k):
        raise ValueError(f"Right shift r ({r}) is not in range 0 <= r < k with k = {k}")
    return sum(c_rj(k, r, j) * us[i - r + j] for j in range(0, k))


# We need to work out the weightd d0, d1, d2 by which the substencils should be multiplied to give
# the big stencil. Since the weights must sum to one, this is a problem of finding the
# convex combination of the small stencils which produce the large stencil.
#
# We can represent the coefficients of each substencil polynomial as a vector and assemble these
# vectors into a matrix so that each row of the matrix corresponds to one cell in the big stencil.
#

# Simple lineaar solver: https://stackoverflow.com/a/31959226/107907

def gamma(k, n):
    h = 1
    return Fraction(
        ((-1)**(n + k) * fac(n)**2),
        ( h**n * fac(n - k) * fac(k) * fac(2*n))
    )



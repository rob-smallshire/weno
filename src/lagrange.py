"""Lagrange Polynomials."""
import operator
from functools import reduce

import sympy
from more_itertools import pairwise

from sympy import symbols, expand, diff, Rational


def product(args):
    return reduce(operator.mul, args, 1)


def lagrange_polynomial(xs, ys, x_symbol=None):
    """A Langrage polynomial function which interpolates the given points.

    Args:
        xs: The x-coordinates of the points.
        ys: The y-coordinate of the points.
        x_symbol: The returned polynomial is a function of this SymPy variable, or 'x'
            if not specified.
    """
    x = x_symbol or symbols("x")
    xs = tuple(xs)
    return sum(y * lagrange_basis(xs, j, x) for j, y in enumerate(ys))


def lagrange_basis(xs, j, x_symbol=None):
    x = x_symbol or symbols("x")
    x_j = xs[j]
    return expand(product((x - x_m) / (x_j - x_m) for x_m in xs if x_m != x_j))


def lagrange_interpolator(ps, x_symbol=None):
    """A Lagrange polynomial which interpolates the given (x, y) points.

    Args:
        ps: An iterable series of points.
        x_symbol: The returned polynomial is a function of this SymPy variable, or 'x'
            if not specified.
    """
    xs, ys = zip(*ps)
    return lagrange_polynomial(xs, ys, x_symbol)


def integrate_averages(xs, ys):
    """A series of integrated values from averages.

        |   y0   |   y1   |   y2   |  ...  |   yn   |
        x0      x1       x2       x3       xn      xn+1

        Y0      Y1       Y2       Y3       Y4      Yn+1

    Args:
        xs: A sequence x values of interval (i.e. cell) boundaries. This sequence must be
            at least one longer than ys.
        ys: A sequence of the average values of y within the corresponding intervals.

    Returns:
        A sequence of (x, Y) where Y is the integral of y at the cell boundaries.
    """
    num_xs = len(xs)
    num_ys = len(ys)
    if not num_xs >= num_ys + 1:
        raise ValueError(
            f"xs with {num_xs} items does not contain at least one more item than ys with {num_ys} items"
        )
    s = 0
    yield xs[0], s
    for (xa, xb), y in zip(pairwise(xs), ys):
        dx = xb - xa
        s += dx * y
        yield xb, s


def lagrange_primitive_polynomial(xs, ys, x_symbol=None):
    """The primitive function (antiderivative) of a function from averages.

        |   y0   |   y1   |   y2   |  ...  |   yn   |
        x0      x1       x2       x3       xn      xn+1

        Y0      Y1       Y2       Y3       Y4      Yn+1

    Args:
        xs: A sequence x values of interval (i.e. cell) boundaries. This sequence must be
            at least one longer than ys.
        ys: A sequence of the average values of y within the corresponding intervals.
        x_symbol: The returned polynomial is a function of this SymPy variable, or 'x'
            if not specified.
    Returns:

    """
    primitive_points = integrate_averages(xs, ys)
    return lagrange_interpolator(primitive_points, x_symbol)


def reconstruction_coefficients(k, r, cell_width=1, ix=None):
    """

    :param k:
    :param r:
    :param cell_width:
    :param ix:
    :return:
    """

    # e.g. k = 3, gives a big stencil of 2k - 1 = 5 cells

    # |   u[j-2]  |   u[j-1]  |   u[j]  |   u[j+1]  |  u[j+2]  |
    # -2         -1           0         1           2          3

    # 0           1           2         3           4          5
    #                         i
    # i is the zero-based index of the central cell in the big stencil
    # so for k = 3, i = 2, which is k - 1
    i = k - 1

    # The stencil is composed of r cells to the left and s cells to
    # the right, symmetrically around central cell, so
    # r = s = (k - 1) // 2

    # cell boundaries over the big stencil [-2, 3] => [-2, 4)
    #
    x = symbols("x")
    dx = symbols("𝛥x")
    xs = [j * dx for j in range(-k + 1, k + 1)]
    us = [symbols(f"u[j{j:+}]") for j in range(-k, k + 1)]  # Check this range!

    # The small stencil with width k can be left-shifted
    # relative to i by up to r cells where 0 <= r < k
    left = i - r
    right = i - r + k

    # We need one more cell boundary than we do cell averages
    small_stencil_xs = xs[left : right + 1]
    small_stencil_us = us[left:right]
    primitive_polynomial = lagrange_primitive_polynomial(small_stencil_xs, small_stencil_us)
    polynomial = primitive_polynomial.diff(x, 1)

    # In this cell-boundary aligned coordinate system i + 1
    # is in what we would be the i + ½ position in the cell-centered
    # coordinate system. We use a dx (cell size) of one.
    solution = polynomial.subs(x, xs[i + 1]).subs(dx, cell_width).expand()
    coefficients = [solution.coeff(u) for u in small_stencil_us]
    return coefficients


from pprint import pprint


def linear_weights(k):
    """Compute the ideal linear weights which combine small stencils to the large stencil.

    The weights give a convex combination of small stencils which is equivalent to the large
    stencil.
    """
    # The coefficients of each shifted small stencil
    small_stencils_coefficients = [reconstruction_coefficients(k, r) for r in range(k)]
    pprint(small_stencils_coefficients)

    large_stencil_coefficients = reconstruction_coefficients(2 * k - 1, k - 1)
    pprint(large_stencil_coefficients)

    # A weight d_r is associated with each small stencil that is shifted by r places left
    weights = [symbols(f"d_{r}]") for r in range(k)]
    pprint(weights)

    # There are k small stencils, and hence k unknown ideal weights, so we need a system
    # of at least k equations. We have (2*k - 1) available equations, so the system is
    # overdetermined. Each equation gives the one of the large stencil coefficients
    # (i.e. for one cell boundary) as the sum of the products of each small stencil linear weights
    # with the corresponding small stencil coefficient for the same cell boundary.

    # j is the index into the cells of the large stencil
    # r is the right shift of the small stencil, and is used to locate a small stencil weights
    # (weights[r]) or the small stencil coefficients (small_stencil_coefficients[r]).
    equations = [
        sum(
            weights[r] * small_stencils_coefficients[r][r - (k - 1) + j]
            for r in range(max(0, (k - 1) - j), min(k - 1, 2 * (k - 1) - j) + 1)
        )
        - large_stencil_coefficients[j]
        for j in range(2 * k - 1)
    ]

    # This system of equations is overdetermined, so slice off the first k equations (though all
    # equations can be used when working with exact number types such as Rational). Consider using
    # a least-squares solver for inexact types.
    solution = sympy.solve(equations[:k], weights)
    return solution


def smoothness_indicators(k, r):
    """The smoothness indicator for a small stencil with size k, and a left-shift of r.

    Args:
        k: The size of the small stencil.
        r: The left-shift of the small stencil.

    Returns:
        An indicator of the smoothness of the reconstructed function within the stencil.
    """
    x = symbols("x")
    dx = symbols("𝛥x")
    xs = [j * dx for j in range(-k + 1, k + 1)]
    us = [symbols(f"u[j{j:+}]") for j in range(-k+1, k)]

    # i is the zero-based index of the central cell in the big stencil
    # so for k = 3, i = 2, which is k - 1
    i = k - 1

    # The small stencil with width k can be left-shifted
    # relative to i by up to r cells where 0 <= r < k
    left = i - r
    right = i - r + k

    # We need one more cell boundary than we do cell averages
    small_stencil_xs = xs[left : right + 1]
    small_stencil_us = us[left:right]
    primitive_polynomial = lagrange_primitive_polynomial(small_stencil_xs, small_stencil_us)
    polynomial = primitive_polynomial.diff(x, 1)

    # The sum of the squares of scaled L2 norms for all the derivatives of the
    # interpolation polynomial pr(x) over the interval (x_{i − 1/2} , x_{i + 1/2} )
    beta = sympy.Integer(0)
    for l in range(1, k):
        lth_squared_derivative = sympy.diff(polynomial, x, l)**2
        integral_squared_derivative = lth_squared_derivative.as_poly(x).integrate(x)
        lower_limit = integral_squared_derivative.subs(x, xs[k - 1])
        upper_limit = integral_squared_derivative.subs(x, xs[k])
        definite_squared_derivative = upper_limit - lower_limit
        scale_factor = dx**(2 * l - 1)
        scaled_definite_squared_derivative = scale_factor * definite_squared_derivative
        beta += scaled_definite_squared_derivative.expand()

    for m in range(k):
        for n in range(m , k):
            q = us[k - 1 - r + m] * us[k - 1 - r + n]
            c = beta.coeff(q)
            print(c)

    return beta

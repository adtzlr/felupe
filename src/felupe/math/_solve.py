# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np


def solve_2d(A, b, solve=np.linalg.solve, **kwargs):
    r"""Solve a linear equation system for two-dimensional unknowns with optional
    elementwise-operating trailing axes.

    Parameters
    ----------
    A : ndarray of shape (M, N, M, N, ...)
        The left-hand side array of the equation system.
    b : ndarray of shape (M, N, ...)
        The right-hand side array of the equation system.
    solve : callable, optional
        A function with a compatible signature to :func:`numpy.linalg.solve`
        (default is :func:`numpy.linalg.solve`).
    **kwargs : dict, optional
        Optional keyword arguments are passed to the callable ``solve(A, b, **kwargs)``.

    Returns
    -------
    x : ndarray of shape (M, N, ...)
        The solved array of unknowns.

    Notes
    -----
    The first two axes of the rhs ``b`` and the first four axes of the lhs ``A`` are the
    tensor dimensions and all remaining trailing axes are treated as batch dimensions.
    This function finds :math:`x_{kl}` for Eq. :eq:`solve-system`.

    ..  math::
        :label: solve-system

        A_{ijkl} : x_{kl} = b_{ij}

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> np.random.seed(855436)
    >>>
    >>> A = np.random.rand(3, 3, 3, 3, 2, 4)
    >>> b = np.ones((3, 3, 2, 4))
    >>>
    >>> x = fem.math.solve_2d(A, b)

    >>> x[..., 0, 0]
    array([[ 1.1442917 ,  2.14516919,  2.00237954],
           [-0.69463749, -2.46685827, -7.21630899],
           [ 4.44825615,  1.35899745,  1.08645703]])

    >>> x.shape
    (3, 3, 2, 4)

    """

    shape = b.shape[:2]
    size = np.prod(shape)
    trax = b.shape[2:]

    # flatten and reshape A to a 2d-matrix of shape (..., M * N, M * N) and
    # b to a 1d-vector of shape (..., M * N)
    b_1d = np.einsum("i...->...i", b.reshape(size, np.prod(trax)))
    A_1d = np.einsum("ij...->...ij", A.reshape(size, size, np.prod(trax)))

    # move the batch-dimensions to the back and reshape x
    return np.einsum("i...->...i", solve(A_1d, b_1d, **kwargs)).reshape(*shape, *trax)

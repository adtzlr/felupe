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


def solve_nd(A, b, solve=np.linalg.solve, n=1, **kwargs):
    r"""Solve a linear equation system for n-dimensional unknowns with optional
    elementwise-operating trailing axes.

    Parameters
    ----------
    A : ndarray of shape (M, N, M, N, ...)
        The left-hand side array of the equation system.
    b : ndarray of shape (M, N, ...)
        The right-hand side array of the equation system.
    n : int, optional
        The dimension of the unknowns, must be greater or equal zero (default is 1).
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
    This function finds :math:`x_{kl}` for Eq. :eq:`solve-system-nd`.

    ..  math::
        :label: solve-system-nd

        A_{i...j...} : x_{j...} = b_{i...}

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> np.random.seed(855436)
    >>>
    >>> A = np.random.rand(3, 3, 3, 3, 3, 3, 2, 4)
    >>> b = np.ones((3, 3, 3, 1, 1))
    >>>
    >>> x = fem.math.solve_nd(A, b, n=3)

    >>> x[..., 0, 0]
    array([[[ 0.63499407, -1.32830401, -1.14593   ],
            [ 1.38045295, -1.03504171,  0.02272754],
            [-0.40106257,  1.6447736 ,  1.87685462]],

           [[ 0.22602567,  0.53675077,  0.37876301],
            [-0.5807506 ,  1.20191896,  0.46334835],
            [-0.90352229,  0.45065787, -0.53026653]],

           [[ 0.3551602 , -1.06411506,  0.01788346],
            [-1.33151311,  1.76817965,  0.00544784],
            [ 0.67065754,  0.43405431, -1.67714269]]])

    >>> x.shape
    (3, 3, 3, 2, 4)

    """

    # broadcast matrix-axes of lhs
    Ashape_new = list(A.shape)
    Ashape_new[:n] = Ashape_new[n : 2 * n] = np.broadcast_shapes(
        A.shape[:n], A.shape[n : 2 * n]
    )
    A = np.broadcast_to(A, Ashape_new)

    # broadcast matrix-axes of rhs
    bshape_new = list(b.shape)
    bshape_new[:n] = np.broadcast_shapes(A.shape[:n], b.shape[:n])
    b = np.broadcast_to(b, bshape_new)

    # trailing (batch) axes of broadcasted output
    shape = b.shape[:n]
    size = np.prod(shape, dtype=int)
    trax = np.broadcast_shapes(b.shape[n:], A.shape[2 * n :])

    # flatten and reshape A to a 2d-matrix of shape (..., M * N * ..., M * N * ...) and
    # b to a 1d-vector of shape (..., M * N * ..., 1) with batches at leading axis
    b_1d = np.einsum("ik...->...ik", b.reshape(size, 1, -1))
    A_1d = np.einsum("ij...->...ij", A.reshape(size, size, -1))

    # move the batch-dimensions to the back and reshape x
    return np.einsum("...ik->ik...", solve(A_1d, b_1d, **kwargs)).reshape(*shape, *trax)


def solve_2d(A, b, solve=np.linalg.solve, **kwargs):
    r"""Solve a linear equation system for two-dimensional unknowns with optional
    elementwise-operating trailing axes.

    Parameters
    ----------
    A : ndarray of shape (M, N, M, N, ...)
        The left-hand side array of the equation system.
    b : ndarray of shape (M, N, ...)
        The right-hand side array of the equation system.
    n : int, optional
        The dimension of the unknowns (default is 1).
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
    This function finds :math:`x_{kl}` for Eq. :eq:`solve-system-2d`.

    ..  math::
        :label: solve-system-2d

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
    return solve_nd(A=A, b=b, solve=solve, n=2, **kwargs)

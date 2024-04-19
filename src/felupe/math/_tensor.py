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

from collections import namedtuple

import numpy as np

try:
    from einsumt import einsumt
except ModuleNotFoundError:
    from numpy import einsum as einsumt


def identity(A=None, dim=None, shape=None):
    r"""Return identity matrices with ones on the diagonal of the first two axes and
    zeros elsewhere. 
    
    Parameters
    ----------
    A : ndarray of shape (N, M, ...) or None, optional
        The array of input matrices. If provided, the dimension and the shape of the
        trailing axes are taken from this array. If None, ``dim`` and ``shape`` are
        required. Default is None.
    dim : int or None, optional
        The dimension of the matrix axes. If None, it is taken from ``A`` (default is
        None).
    shape : tuple of int or None, optional
        A tuple containing the shape of the trailing axes (batch dimensions). Default is
        None.
    
    Returns
    -------
    ndarray of shape (N, M, *np.ones_like(...)) or (dim, dim, *np.ones_like(...))
        The identity matrix.
    
    Notes
    -----
    The first two axes are the matrix dimensions and all remaining trailing axes are
    treated as batch dimensions. As the identity matrix is idependent of the input
    matrices ``A``, the size of the trailing axes is reduced to one.
    
    The identity matrix is defined by Eq. :eq:`math-identity`
    
    ..  math::
        :label: math-identity
        
        \boldsymbol{1} = \boldsymbol{A} \boldsymbol{A}^{-1} 
            = \boldsymbol{A}^{-1} \boldsymbol{A}
    
    and it is shown in Eq. :eq:`math-identity-items`.
    
    ..  math::
        :label: math-identity-items
        
        \boldsymbol{1} = \begin{bmatrix}
                1 & 0 & 0 \\
                0 & 1 & 0 \\
                0 & 0 & 1
            \end{bmatrix}
    
    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>> 
    >>> A = np.random.rand(3, 2, 8, 20)
    >>> I = fem.math.identity(A)
    >>> I.shape
    (3, 2, 1, 1)
    
    With given dimension of the matrix axes the shape of the output is different.
    
    >>> fem.math.identity(A, dim=2).shape
    (2, 2, 1, 1)
    
    Note how the number of batch axes change if a ``shape`` is given.
    
    >>> fem.math.identity(A, shape=(4, 7, 3)).shape
    (3, 2, 1, 1, 1)
    
    See Also
    --------
    numpy.eye : Return a 2-D array with ones on the diagonal and zeros elsewhere.
    """

    M = None
    if A is not None:
        N, M = A.shape[:2]
        shapeA = A.shape[2:]
        if dim is None:
            dim = N
        if shape is None:
            shape = shapeA

    ones = (1,) * len(shape)
    eye = np.eye(N=dim, M=M)
    return eye.reshape(*eye.shape, *ones)


def sym(A, out=None):
    r"""Return the symmetric parts of second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (M, M, ...)
        The array of second-order tensors.
    out : ndarray or None, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ndarray of shape (M, M, ...)
        Array with the symmetric parts of the second-order tensors.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The symmetric part of a second-order tensor is obtained by Eq. :eq:`math-symmetric`.

    ..  math::
        :label: math-symmetric

        \text{sym} \left( \boldsymbol{A} \right) = \frac{1}{2} \left(
                \boldsymbol{A} + \boldsymbol{A}^T
            \right)

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(18, dtype=float).reshape(2, 3, 3).T)
    >>> A[..., 0]
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])

    >>> fem.math.sym(A)[..., 0]
    array([[0., 2., 4.],
           [2., 4., 6.],
           [4., 6., 8.]])

    """
    out = np.add(A, transpose(A), out=out)
    return np.multiply(out, 0.5, out=out)


def dya(A, B, mode=2, parallel=False, **kwargs):
    r"""Return the dyadic product of two first-order or two second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (N, ...) or (N, N, ...)
        Array with first-order or second-order tensors.
    B : ndarray of shape (M, ...) or (M, M, ...)
        Array with first-order or second-order tensors.
    mode : int, optional
        Mode of operation. Return the dyadic products of two second-order tensors with
        2 and the dyadic products of two first-order tensors with 1. Default is 2.
    parallel : bool, optional
        A flag to enable a threaded evaluation of the results (default is False).
    **kwargs : dict, optional
        Optional keyword-arguments for :func:`numpy.multiply`, e.g. ``out=None``.

    Returns
    -------
    ndarray of shape (N, M, ...) or (N, N, M, M, ...)
        The array of dyadic products.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions. The definition of the dyadic product is given in Eq.
    :eq:`math-dya2` for two second-order tensors

    ..  math::
        :label: math-dya2

        \mathbb{C} &= \boldsymbol{A} \otimes \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{ij}\ B_{kl}

    and in Eq. :eq:`math-dya1` for two first-order tensors.

    ..  math::
        :label: math-dya1

        \boldsymbol{C} &= \boldsymbol{a} \otimes \boldsymbol{b}

        C_{ij} &= a_{i} \otimes b_{j}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T)
    >>> B = fem.math.transpose(np.arange(100, 109, dtype=float).reshape(1, 3, 3).T)
    >>> C = fem.math.dya(A, B)
    >>> C.shape
    (3, 3, 3, 3, 1)

    >>> C[..., 0].reshape(9, 9)
    array([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
           [100., 101., 102., 103., 104., 105., 106., 107., 108.],
           [200., 202., 204., 206., 208., 210., 212., 214., 216.],
           [300., 303., 306., 309., 312., 315., 318., 321., 324.],
           [400., 404., 408., 412., 416., 420., 424., 428., 432.],
           [500., 505., 510., 515., 520., 525., 530., 535., 540.],
           [600., 606., 612., 618., 624., 630., 636., 642., 648.],
           [700., 707., 714., 721., 728., 735., 742., 749., 756.],
           [800., 808., 816., 824., 832., 840., 848., 856., 864.]])

    See Also
    --------
    felupe.math.cdya : Crossed-dyadic product of two second-order tensors.
    felupe.math.cdya_ik : ik-crossed dyadic product of two second-order tensors.
    felupe.math.cdya_il : il-crossed dyadic product of two second-order tensors.

    """

    if mode == 2:
        return np.multiply(A[:, :, None, None], B[None, None, :, :], **kwargs)
    elif mode == 1:
        return np.multiply(A[:, None], B[None, :], **kwargs)
    else:
        raise ValueError("unknown mode. (1 or 2)", mode)


def inv(A, determinant=None, full_output=False, sym=False, out=None):
    r"""Return the inverses of second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (1, 1, ...), (2, 2, ...) or (3, 3, ...)
        The array of second-order tensors.
    determinant : ndarray or None, optional
        The array with the pre-evaluated determinants of the second-order tensors (
        default is None).
    full_output : bool, optional
        A flag to return the array of inverses and the determinants (default is False).
    sym : bool, optional
        A flag to assume symmetric second-order tensors. Only the upper-triangle
        elements are taken into account. Default is False.
    out : ndarray or None, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ndarray of shape (1, 1, ...), (2, 2, ...) or (3, 3, ...)
        The inverses of second-order tensors.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The inverse of a three-dimensional second-order tensor is obtained by Eq.
    :eq:`math-inv` and Eq. :eq:`math-inv-matrix`

    ..  math::
        :label: math-inv

        \boldsymbol{A}^{-1} \boldsymbol{A} &= \boldsymbol{A} \boldsymbol{A}^{-1} =
            \boldsymbol{1}

        \boldsymbol{A}^{-1} &= \frac{1}{\det(\boldsymbol{A})}
            \text{cof}(\boldsymbol{A})^T
    
    ..  math::
        :label: math-inv-matrix

        \boldsymbol{A}^{-1} = \frac{1}{\det(\boldsymbol{A})} \begin{bmatrix}
                \left( \boldsymbol{A}_2 \times \boldsymbol{A}_3 \right)^T \\
                \left( \boldsymbol{A}_3 \times \boldsymbol{A}_1 \right)^T \\
                \left( \boldsymbol{A}_1 \times \boldsymbol{A}_2 \right)^T
            \end{bmatrix}

    with the column (grid) vectors :math:`\boldsymbol{A}_j`, see Eq. :eq:`math-gridvec`.

    ..  math::
        :label: math-gridvec

        \boldsymbol{A} = \begin{bmatrix}
                \boldsymbol{A}_1 & \boldsymbol{A}_2 & \boldsymbol{A}_3
            \end{bmatrix}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3, 1)
    >>> A[..., 0]
    array([[1. , 1.3, 1.5],
           [1.3, 1.1, 1.4],
           [1.5, 1.4, 1.2]])
    
    >>> invA = fem.math.inv(A)
    >>> invA[..., 0]
    array([[-2.01892744,  1.70347003,  0.5362776 ],
           [ 1.70347003, -3.31230284,  1.73501577],
           [ 0.5362776 ,  1.73501577, -1.86119874]])
    
    >>> fem.math.dot(A, invA)[..., 0].round(3)
    array([[ 1., -0.,  0.],
           [-0.,  1.,  0.],
           [-0., -0.,  1.]])

    See Also
    --------
    felupe.math.det : Return the determinants of second order tensors.
    felupe.math.cof : Return the cofactors of second order tensors.
    """

    detAinvA = out
    if detAinvA is None:
        detAinvA = np.zeros_like(A)

    x1 = None
    x2 = None
    if A.shape[:2] == (3, 3):
        # diagonal items
        x1 = np.multiply(A[1, 2], A[2, 1], out=x1)
        x2 = np.multiply(A[1, 1], A[2, 2], out=x2)
        np.add(-x1, x2, out=detAinvA[0, 0])

        x1 = np.multiply(A[0, 2], A[2, 0], out=x1)
        x2 = np.multiply(A[0, 0], A[2, 2], out=x2)
        np.add(-x1, x2, out=detAinvA[1, 1])

        x1 = np.multiply(A[0, 1], A[1, 0], out=x1)
        x2 = np.multiply(A[0, 0], A[1, 1], out=x2)
        np.add(-x1, x2, out=detAinvA[2, 2])

        # upper-triangle off-diagonal
        x1 = np.multiply(A[0, 1], A[2, 2], out=x1)
        x2 = np.multiply(A[0, 2], A[2, 1], out=x2)
        np.add(-x1, x2, out=detAinvA[0, 1])

        x1 = np.multiply(A[0, 2], A[1, 1], out=x1)
        x2 = np.multiply(A[0, 1], A[1, 2], out=x2)
        np.add(-x1, x2, out=detAinvA[0, 2])

        x1 = np.multiply(A[0, 0], A[1, 2], out=x1)
        x2 = np.multiply(A[0, 2], A[1, 0], out=x2)
        np.add(-x1, x2, out=detAinvA[1, 2])

        if sym:
            detAinvA[1, 0] = detAinvA[0, 1]
            detAinvA[2, 0] = detAinvA[0, 2]
            detAinvA[2, 1] = detAinvA[1, 2]
        else:
            # lower-triangle off-diagonal
            x1 = np.multiply(A[1, 0], A[2, 2], out=x1)
            x2 = np.multiply(A[2, 0], A[1, 2], out=x2)
            np.add(-x1, x2, out=detAinvA[1, 0])

            x1 = np.multiply(A[2, 0], A[1, 1], out=x1)
            x2 = np.multiply(A[1, 0], A[2, 1], out=x2)
            np.add(-x1, x2, out=detAinvA[2, 0])

            x1 = np.multiply(A[0, 0], A[2, 1], out=x1)
            x2 = np.multiply(A[2, 0], A[0, 1], out=x2)
            np.add(-x1, x2, out=detAinvA[2, 1])

    elif A.shape[:2] == (2, 2):
        detAinvA[0, 0] = A[1, 1]
        detAinvA[0, 1] = -A[0, 1]
        detAinvA[1, 0] = -A[1, 0]
        detAinvA[1, 1] = A[0, 0]

    elif A.shape[:2] == (1, 1):
        detAinvA[0, 0] = 1

    else:
        raise ValueError(
            " ".join(
                [
                    "Wrong shape of first two axes.",
                    "Must be (1, 1), (2, 2) or (3, 3) but {A.shape[:2]} is given.",
                ]
            )
        )

    if determinant is None:
        detA = det(A, out=x1)
    else:
        detA = determinant

    if full_output:
        return np.divide(detAinvA, detA, out=detAinvA), detA
    else:
        return np.divide(detAinvA, detA, out=detAinvA)


def det(A, out=None):
    r"""Return the determinants of second order tensors.

    Parameters
    ----------
    A : ndarray of shape (M, M, ...)
        The array of second-order tensors.
    out : ndarray or None, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ndarray of shape (...)
        Array with the determinants.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The determinant of a second-order tensor is obtained by Eq. :eq:`math-det`.

    ..  math::
        :label: math-det

        \det \left( \boldsymbol{A} \right) &=
              A_{11}~A_{22}~A_{33}
            + A_{12}~A_{23}~A_{31}
            + A_{13}~A_{21}~A_{32}

           &- A_{31}~A_{22}~A_{13}
            - A_{32}~A_{23}~A_{11}
            - A_{33}~A_{21}~A_{12}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T) / 10
    >>> A += np.eye(3).reshape(3, 3, 1)

    >>> A.shape
    (3, 3, 1)

    >>> A[..., 0]
    array([[1. , 0.1, 0.2],
           [0.3, 1.4, 0.5],
           [0.6, 0.7, 1.8]])

    >>> fem.math.det(A)[..., 0]
    array(2.02)

    """

    detA = out
    if detA is None:
        detA = np.zeros_like(A[0, 0])
    else:
        detA.fill(0)

    if A.shape[:2] == (3, 3):
        tmp = np.multiply(A[0, 0], A[1, 1])
        np.multiply(tmp, A[2, 2], out=tmp)
        np.add(detA, tmp, out=detA)

        tmp = np.multiply(A[0, 1], A[1, 2], out=tmp)
        np.multiply(tmp, A[2, 0], out=tmp)
        np.add(detA, tmp, out=detA)

        tmp = np.multiply(A[0, 2], A[1, 0], out=tmp)
        np.multiply(tmp, A[2, 1], out=tmp)
        np.add(detA, tmp, out=detA)

        tmp = np.multiply(A[2, 0], A[1, 1], out=tmp)
        np.multiply(tmp, A[0, 2], out=tmp)
        np.add(detA, -tmp, out=detA)

        tmp = np.multiply(A[2, 1], A[1, 2], out=tmp)
        np.multiply(tmp, A[0, 0], out=tmp)
        np.add(detA, -tmp, out=detA)

        tmp = np.multiply(A[2, 2], A[1, 0], out=tmp)
        np.multiply(tmp, A[0, 1], out=tmp)
        np.add(detA, -tmp, out=detA)

    elif A.shape[:2] == (2, 2):
        tmp = np.multiply(A[0, 0], A[1, 1])
        np.add(detA, tmp, out=detA)

        tmp = np.multiply(A[1, 0], A[0, 1], out=tmp)
        np.add(detA, -tmp, out=detA)

    elif A.shape[:2] == (1, 1):
        np.add(detA, A[0, 0], out=detA)

    else:
        raise ValueError(
            " ".join(
                [
                    "Wrong shape of first two axes.",
                    f"Must be (1, 1), (2, 2) or (3, 3) but {A.shape[:2]} is given.",
                ]
            )
        )
    return detA


def dev(A, out=None):
    r"""Return the deviatoric parts of second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (M, M, ...)
        The array of second-order tensors.
    out : ndarray or None, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ndarray of shape (M, M, ...)
        The deviatoric parts of second-order tensors.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The deviatoric part of a three-dimensional second-order tensor is obtained by Eq.
    :eq:`math-dev`.

    ..  math::
        :label: math-dev

        \text{dev} \left( \boldsymbol{A} \right) = \boldsymbol{A}
            - \frac{\text{tr}(\boldsymbol{A})}{3} \boldsymbol{1}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T)
    >>> A[..., 0]
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])

    >>> fem.math.sym(A)[..., 0]
    array([[0., 2., 4.],
           [2., 4., 6.],
           [4., 6., 8.]])

    See Also
    --------
    numpy.eye : Return a 2-D array with ones on the diagonal and zeros elsewhere.
    """
    dim = A.shape[0]
    return np.add(A, -trace(A) / dim * identity(A), out=out)


def cof(A, sym=False, out=None):
    r"""Return the cofactors of second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (1, 1, ...), (2, 2, ...) or (3, 3, ...)
        The array of second-order tensors.
    sym : bool, optional
        A flag to assume symmetric second-order tensors. Only the upper-triangle
        elements are taken into account. Default is False.
    out : ndarray or None, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ndarray of shape (1, 1, ...), (2, 2, ...) or (3, 3, ...)
        The cofactors of second-order tensors.

    Notes
    -----
    The first two axes are the matrix dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The inverse of a three-dimensional second-order tensor is obtained by Eq.
    :eq:`math-cof`

    ..  math::
        :label: math-cof

        \text{cof}(\boldsymbol{A}) &= \det (\boldsymbol{A}) \boldsymbol{A}^{-T}

        \text{cof}(\boldsymbol{A}) &= \begin{bmatrix}
                \boldsymbol{A}_2 \times \boldsymbol{A}_3 &
                \boldsymbol{A}_3 \times \boldsymbol{A}_1 &
                \boldsymbol{A}_1 \times \boldsymbol{A}_2
            \end{bmatrix}

    with the column (grid) vectors :math:`\boldsymbol{A}_j`, see Eq. :eq:`math-gridv`.

    ..  math::
        :label: math-gridv

        \boldsymbol{A} = \begin{bmatrix}
                \boldsymbol{A}_1 & \boldsymbol{A}_2 & \boldsymbol{A}_3
            \end{bmatrix}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3, 1)
    >>> A[..., 0]
    array([[1. , 1.3, 1.5],
           [1.3, 1.1, 1.4],
           [1.5, 1.4, 1.2]])

    >>> cofA = fem.math.cof(A)
    >>> cofA[..., 0]
    array([[-0.64,  0.54,  0.17],
           [ 0.54, -1.05,  0.55],
           [ 0.17,  0.55, -0.59]])

    >>> (fem.math.det(A) * fem.math.transpose(fem.math.inv(A)))[..., 0]
    array([[-0.64,  0.54,  0.17],
           [ 0.54, -1.05,  0.55],
           [ 0.17,  0.55, -0.59]])

    See Also
    --------
    felupe.math.det : Return the determinants of second order tensors.
    felupe.math.inv : Return the inverses of second order tensors.
    """
    return transpose(inv(A, determinant=1.0, sym=sym, out=out))


def eig(a, eig=np.linalg.eig, **kwargs):
    """Compute the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : ndarray of shape (M, M, ...)
        Matrices for which the eigenvalues and right eigenvectors will be computed.
    eig : callable, optional
        A callable for the eigenvalue and eigenvector evaluation compatible with
        :func:`numpy.linalg.eig` (default is :func:`numpy.linalg.eig`).
    **kwargs : dict, optional
        Optional keyword-arguments are passed to ``eig(a, **kwargs)``.

    Returns
    -------
    A namedtuple with the following attributes:
    eigenvalues : ndarray of shape (M, ...)
        The eigenvalues, each repeated according to its multiplicity. The eigenvalues
        are not necessarily ordered. The resulting array will be of complex type,
        unless the imaginary part is zero in which case it will be cast to a real type.
        When a is real the resulting eigenvalues will be real (0 imaginary part) or
        occur in conjugate pairs.
    eigenvectors : ndarray of shape (M, M, ...)
        The normalized (unit "length") eigenvectors, such that the column
        ``eigenvectors[:, i]`` is the eigenvector corresponding to the eigenvalue
        ``eigenvalues[i]``.

    Notes
    -----
    ..  note::
        The first two axes are the tensor dimensions and all remaining trailing axes are
        treated as batch dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> x = np.array([1, 0, 0.3, 0, 1.3, 0, 0, 0, 0.7]).reshape(3, 3)
    >>> y = np.array([1, 0, 0, 0, 0.4, 0.1, 0, 0, 1.6]).reshape(3, 3)
    >>> F = np.stack([x, y], axis=2)
    >>>
    >>> C = fem.math.dot(fem.math.transpose(F), F)
    >>> w, v = fem.math.eig(C)
    >>>
    >>> w[0]
    array([1.15619667, 2.57066372])

    The associated eigenvectors are extracted. The first column is the eigenvector
    for the first right Cauchy-Green deformation tensor and the second column for the
    second right Cauchy-Green deformation tensor.

    >>> v[:, 0]
    array([[0.88697868, 0.        ],
           [0.        , 0.01659066],
           [0.46181038, 0.99986237]])

    See Also
    --------
    felupe.math.eigh : Return the eigenvalues and eigenvectors of a complex Hermitian
        (conjugate symmetric) or a real symmetric matrix.
    numpy.linalg.eig : Compute the eigenvalues and right eigenvectors of a square array.
    numpy.linalg.eigh : Return the eigenvalues and eigenvectors of a complex Hermitian
        (conjugate symmetric) or a real symmetric matrix.
    """

    res = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
    eigenvalues, eigenvectors = eig(np.einsum("ij...->...ij", a), **kwargs)

    return res(
        eigenvalues=np.einsum("...a->a...", eigenvalues),
        eigenvectors=np.einsum("...ia->ia...", eigenvectors),
    )


def eigh(a, UPLO="L"):
    """Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate
    symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of a, and a 2-D square
    array or matrix (depending on the input type) of the corresponding eigenvectors (in
    columns).

    Parameters
    ----------
    a : ndarray of shape (M, M, ...)
        Matrices for which the eigenvalues and right eigenvectors will be computed.
    UPLO : {"L", "U"}, optional
        Specifies whether the calculation is done with the lower triangular part of `a`
        ('L', default) or the upper triangular part ('U'). Irrespective of this value
        only the real parts of the diagonal will be considered in the computation to
        preserve the notion of a Hermitian matrix. It therefore follows that the
        imaginary part of the diagonal will always be treated as zero.

    Returns
    -------
    A namedtuple with the following attributes:
    eigenvalues : (M, ...) ndarray
        The eigenvalues in ascending order, each repeated according to its multiplicity.
    eigenvectors : (M, M, ...) ndarray
        The normalized (unit "length") eigenvectors, such that the column
        ``eigenvectors[:, i]`` is the eigenvector corresponding to the eigenvalue
        ``eigenvalues[i]``.

    Notes
    -----
    ..  note::
        The first two axes are the tensor dimensions and all remaining trailing axes are
        treated as batch dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> x = np.array([1, 0, 0.3, 0, 1.3, 0, 0, 0, 0.7]).reshape(3, 3)
    >>> y = np.array([1, 0, 0, 0, 0.4, 0.1, 0, 0, 1.6]).reshape(3, 3)
    >>> F = np.stack([x, y], axis=2)
    >>>
    >>> C = fem.math.dot(fem.math.transpose(F), F)
    >>> w, v = fem.math.eigh(C)
    >>>
    >>> w[-1]
    array([1.69      , 2.57066372])

    The associated eigenvectors are extracted. The first column is the eigenvector,
    associated to the greated eigenvalue, for the first right Cauchy-Green deformation
    tensor and the second column for the second right Cauchy-Green deformation tensor.

    >>> v[:, 0]
    array([[-4.61810381e-01,  0.00000000e+00],
           [ 1.11022302e-16, -9.99862366e-01],
           [ 8.86978676e-01,  1.65906569e-02]])


    See Also
    --------
    felupe.math.eig : Compute the eigenvalues and right eigenvectors of a square array.
    numpy.linalg.eig : Compute the eigenvalues and right eigenvectors of a square array.
    numpy.linalg.eigh : Return the eigenvalues and eigenvectors of a complex Hermitian
        (conjugate symmetric) or a real symmetric matrix.
    """
    return eig(a, eig=np.linalg.eigh)


def eigvals(a, shear=False, eigvals=np.linalg.eigvals, **kwargs):
    "Eigenvalues (and optional principal shear values) of a matrix A."
    eigenvalues = eigvals(a.T, **kwargs).T
    if shear:
        dim = eigenvalues.shape[0]
        if dim == 3:
            ij = [(1, 0), (2, 0), (2, 1)]
        elif dim == 2:
            ij = [(1, 0)]
        eigenvalues_diff = np.array([eigenvalues[i] - eigenvalues[j] for i, j in ij])
        return np.vstack((eigenvalues, eigenvalues_diff))
    else:
        return eigenvalues


def eigvalsh(A, shear=False):
    "Eigenvalues (and optional principal shear values) of a symmetric matrix A."
    return eigvals(A, shear=shear, eigvals=np.linalg.eigvalsh)


def transpose(A, mode=1):
    "Transpose (mode=1) or major-transpose (mode=2) of matrix A."
    if mode == 1:
        return np.einsum("ij...->ji...", A)
    elif mode == 2:
        return np.einsum("ijkl...->klij...", A)
    else:
        raise ValueError("Unknown value of mode.")


def majortranspose(A):
    return transpose(A, mode=2)


def trace(A, out=None):
    r"""Return the sums along the diagonals of second order tensors.

    Parameters
    ----------
    A : ndarray of shape (M, M, ...)
        The array of second-order tensors.
    out : ndarray or None, optional
        If provided, the calculation is done into this array.

    Returns
    -------
    ndarray of shape (...)
        Array with the sums along the diagonals.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The trace of a second-order tensor is obtained by Eq. :eq:`math-trace`.

    ..  math::
        :label: math-trace

        \text{tr} \left( \boldsymbol{A} \right) &= \boldsymbol{A} : \boldsymbol{1}

        A_{ii} &= A_{ij} : \delta_{ij}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T)
    >>> A.shape
    (3, 3, 1)

    >>> A[..., 0]
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])

    >>> fem.math.trace(A)[..., 0]
    array(12.)

    See Also
    --------
    numpy.trace : Return the sum along diagonals of the array.

    """
    return np.trace(A, out=out)


def cdya_ik(A, B, parallel=False, **kwargs):
    r"""Return the ik-crossed dyadic product of two second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (N, N, ...)
        Array with second-order tensors.
    B : ndarray of shape (M, M, ...)
        Array with second-order tensors.
    parallel : bool, optional
        A flag to enable a threaded evaluation of the results (default is False).
    **kwargs : dict, optional
        Optional keyword-arguments for :func:`numpy.einsum`, e.g. ``out=None``.

    Returns
    -------
    ndarray of shape  (N, M, N, M, ...)
        The array of ik-crossed dyadic products.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions. The definition of the ik-crossed dyadic product is
    given in Eq. :eq:`math-cdya-ik`.

    ..  math::
        :label: math-cdya-ik

        \mathbb{C} &= \boldsymbol{A} \overset{ik}{\otimes} \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{ik}\ B_{jl}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T)
    >>> B = fem.math.transpose(np.arange(100, 109, dtype=float).reshape(1, 3, 3).T)
    >>> C = fem.math.cdya_ik(A, B)
    >>> C.shape
    (3, 3, 3, 3, 1)

    >>> C[..., 0].reshape(9, 9)
    array([[  0.,   0.,   0., 100., 101., 102., 200., 202., 204.],
           [  0.,   0.,   0., 103., 104., 105., 206., 208., 210.],
           [  0.,   0.,   0., 106., 107., 108., 212., 214., 216.],
           [300., 303., 306., 400., 404., 408., 500., 505., 510.],
           [309., 312., 315., 412., 416., 420., 515., 520., 525.],
           [318., 321., 324., 424., 428., 432., 530., 535., 540.],
           [600., 606., 612., 700., 707., 714., 800., 808., 816.],
           [618., 624., 630., 721., 728., 735., 824., 832., 840.],
           [636., 642., 648., 742., 749., 756., 848., 856., 864.]])

    See Also
    --------
    felupe.math.dya : Dyadic product of two first-order or two second-order tensors.
    felupe.math.cdya : Crossed dyadic product of two second-order tensors.
    felupe.math.cdya_il : il-crossed dyadic product of two second-order tensors.

    """
    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum
    return einsum("ij...,kl...->ikjl...", A, B, **kwargs)


def cdya_il(A, B, parallel=False, **kwargs):
    r"""Return the il-crossed dyadic product of two second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (N, N, ...)
        Array with second-order tensors.
    B : ndarray of shape (M, M, ...)
        Array with second-order tensors.
    parallel : bool, optional
        A flag to enable a threaded evaluation of the results (default is False).
    **kwargs : dict, optional
        Optional keyword-arguments for :func:`numpy.einsum`, e.g. ``out=None``.

    Returns
    -------
    ndarray of shape  (N, M, N, M, ...)
        The array of ik-crossed dyadic products.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions. The definition of the il-crossed dyadic product is
    given in Eq. :eq:`math-cdya-il`.

    ..  math::
        :label: math-cdya-il

        \mathbb{C} &= \boldsymbol{A} \overset{il}{\otimes} \boldsymbol{B}

        \mathbb{C}_{ijkl} &= A_{il}\ B_{kj}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T)
    >>> B = fem.math.transpose(np.arange(100, 109, dtype=float).reshape(1, 3, 3).T)
    >>> C = fem.math.cdya_il(A, B)
    >>> C.shape
    (3, 3, 3, 3, 1)

    >>> C[..., 0].reshape(9, 9)
    array([[  0., 100., 200.,   0., 103., 206.,   0., 106., 212.],
           [  0., 101., 202.,   0., 104., 208.,   0., 107., 214.],
           [  0., 102., 204.,   0., 105., 210.,   0., 108., 216.],
           [300., 400., 500., 309., 412., 515., 318., 424., 530.],
           [303., 404., 505., 312., 416., 520., 321., 428., 535.],
           [306., 408., 510., 315., 420., 525., 324., 432., 540.],
           [600., 700., 800., 618., 721., 824., 636., 742., 848.],
           [606., 707., 808., 624., 728., 832., 642., 749., 856.],
           [612., 714., 816., 630., 735., 840., 648., 756., 864.]])

    See Also
    --------
    felupe.math.dya : Dyadic product of two first-order or two second-order tensors.
    felupe.math.cdya_ik : ik-crossed dyadic product of two second-order tensors.
    felupe.math.cdya : Crossed dyadic product of two second-order tensors.

    """
    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum
    return einsum("ij...,kl...->ilkj...", A, B, **kwargs)


def cdya(A, B, parallel=False, out=None, **kwargs):
    r"""Return the crossed dyadic product of two second-order tensors.

    Parameters
    ----------
    A : ndarray of shape (M, M, ...)
        Array with second-order tensors.
    B : ndarray of shape (M, M, ...)
        Array with second-order tensors.
    parallel : bool, optional
        A flag to enable a threaded evaluation of the results (default is False).
    **kwargs : dict, optional
        Optional keyword-arguments for :func:`numpy.einsum`, e.g. ``out=None``.

    Returns
    -------
    ndarray of shape  (M, M, M, M, ...)
        The array of ik-crossed dyadic products.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions. The definition of the crossed dyadic product is
    given in Eq. :eq:`math-cdya`.

    ..  math::
        :label: math-cdya

        \mathbb{C} &= \boldsymbol{A} \odot \boldsymbol{B} = \frac{1}{2} \left(
            \boldsymbol{A} \overset{ik}{\otimes} \boldsymbol{B} +
            \boldsymbol{A} \overset{il}{\otimes} \boldsymbol{B}
        \right)

        \mathbb{C}_{ijkl} &= \frac{1}{2} \left( A_{ik}~B_{jl} + A_{il}~B_{kj} \right)

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T)
    >>> B = fem.math.transpose(np.arange(100, 109, dtype=float).reshape(1, 3, 3).T)
    >>> C = fem.math.cdya(A, B)
    >>> C.shape
    (3, 3, 3, 3, 1)

    >>> C[..., 0].reshape(9, 9)
    array([[  0. ,  50. , 100. ,  50. , 102. , 154. , 100. , 154. , 208. ],
           [  0. ,  50.5, 101. ,  51.5, 104. , 156.5, 103. , 157.5, 212. ],
           [  0. ,  51. , 102. ,  53. , 106. , 159. , 106. , 161. , 216. ],
           [300. , 351.5, 403. , 354.5, 408. , 461.5, 409. , 464.5, 520. ],
           [306. , 358. , 410. , 362. , 416. , 470. , 418. , 474. , 530. ],
           [312. , 364.5, 417. , 369.5, 424. , 478.5, 427. , 483.5, 540. ],
           [600. , 653. , 706. , 659. , 714. , 769. , 718. , 775. , 832. ],
           [612. , 665.5, 719. , 672.5, 728. , 783.5, 733. , 790.5, 848. ],
           [624. , 678. , 732. , 686. , 742. , 798. , 748. , 806. , 864. ]])

    See Also
    --------
    felupe.math.dya : Dyadic product of two first-order or two second-order tensors.
    felupe.math.cdya_ik : ik-crossed dyadic product of two second-order tensors.
    felupe.math.cdya_il : il-crossed dyadic product of two second-order tensors.

    """
    res = cdya_ik(A, B, parallel=parallel, out=out, **kwargs)
    res = np.add(res, cdya_il(A, B, parallel=parallel, **kwargs), out=res)
    return np.multiply(res, 0.5, out=res)


def cross(a, b):
    r"""Return the cross product of two vectors.

    Parameters
    ----------
    a : ndarray of shape (N, ...)
        First array of vectors.
    b : ndarray of shape (N, ...)
        Second array of vectors.

    Returns
    -------
    c : ndarray of shape (N, ...)
        Vector cross-products.

    Notes
    -----
    The first axis is the vector dimension and all remaining trailing axes are
    treated as batch dimensions.

    .. math::

       \boldsymbol{c} = \boldsymbol{a} \times \boldsymbol{b}

    .. note::
       This is :func:`numpy.cross` with ``axisa = axisb = axisc = 0``.

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> a = np.arange(30, dtype=float).reshape(3, 2, 5)
    >>> b = np.arange(80, 20, -2, dtype=float).reshape(5, 2, 3).T
    >>> fem.math.cross(a, b)
    array([[[-800., -682., -564., -446., -328.],
            [-750., -632., -514., -396., -278.]],
    <BLANKLINE>
           [[1600., 1364., 1128.,  892.,  656.],
            [1500., 1264., 1028.,  792.,  556.]],
    <BLANKLINE>
           [[-800., -682., -564., -446., -328.],
            [-750., -632., -514., -396., -278.]]])

    See Also
    --------
    numpy.cross : Return the cross product of two (arrays of) vectors.
    """
    return np.cross(a, b, axisa=0, axisb=0, axisc=0)


def dot(A, B, mode=(2, 2), parallel=False, **kwargs):
    "Dot-product of A and B with inputs of n trailing axes.."

    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum

    if mode == (2, 2):
        return einsum("ik...,kj...->ij...", A, B, **kwargs)
    elif mode == (1, 1):
        return einsum("i...,i...->...", A, B, **kwargs)
    elif mode == (4, 4):
        return einsum("ijkp...,plmn...->ijklmn...", A, B, **kwargs)
    elif mode == (2, 1):
        return einsum("ij...,j...->i...", A, B, **kwargs)
    elif mode == (1, 2):
        return einsum("i...,ij...->j...", A, B, **kwargs)
    elif mode == (2, 3):
        return einsum("im...,mjk...->ijk...", A, B, **kwargs)
    elif mode == (3, 2):
        return einsum("ijm...,mk...->ijk...", A, B, **kwargs)
    elif mode == (4, 1):
        return einsum("ijkl...,l...->ijk...", A, B, **kwargs)
    elif mode == (1, 4):
        return einsum("i...,ijkl...->jkl...", A, B, **kwargs)
    elif mode == (2, 4):
        return einsum("im...,mjkl...->ijkl...", A, B, **kwargs)
    elif mode == (4, 2):
        return einsum("ijkm...,ml...->ijkl...", A, B, **kwargs)
    else:
        raise TypeError("Unknown shape of A and B.")


def ddot(A, B, mode=(2, 2), parallel=False, **kwargs):
    r"""Return the double-dot products of first-, second-, third- or fourth-order
    tensors.

    Parameters
    ----------
    A : ndarray of shape (M, M, ...)
        Array with first-, second-, third- or fourth-order tensors.
    B : ndarray of shape (M, M, ...)
        Array with first-, second-, third- or fourth-order tensors.
    mode : tuple of int, optional
        Mode of operation. Return the double-dot products of two second-order tensors
        with (2, 2), of a second-order and a fourth-order tensor with (2, 4) and so on.
        Default is (2, 2).
    parallel : bool, optional
        A flag to enable a threaded evaluation of the results (default is False).
    **kwargs : dict, optional
        Optional keyword-arguments for :func:`numpy.einsum`, e.g. ``out=None``.

    Returns
    -------
    ndarray of shape (...)
        Array with the double-dot products.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The double-dot product is obtained by Eq. :eq:`math-ddot22` for two second-order
    tensors,

    ..  math::
        :label: math-ddot22

        c &= \boldsymbol{A} : \boldsymbol{B}

        c &= A_{ij} : B_{ij}

    by Eq. :eq:`math-ddot44` for two fourth-order tensors,

    ..  math::
        :label: math-ddot44

        \mathbb{C} &= \mathbb{A} : \mathbb{B}

        \mathbb{C}_{ijmn} &=  \mathbb{A}_{ijkl} : \mathbb{B}_{klmn}

    by Eq. :eq:`math-ddot23` for a second-order and a third-order tensor,

    ..  math::
        :label: math-ddot23

        \boldsymbol{c} = \boldsymbol{A} : \mathcal{B}

        \qquad c_{k} &= A_{ij} : \mathcal{B}_{ijk}

    by Eq. :eq:`math-ddot32` for a third-order and a second-order tensor,

    ..  math::
        :label: math-ddot32

        \boldsymbol{c} = \mathcal{A} : \boldsymbol{A}

        \qquad c_{i} &= \mathcal{A}_{ijk} : B_{jk}

    by Eq. :eq:`math-ddot24` for a second-order and a fourth-order tensor

    ..  math::
        :label: math-ddot24

        \boldsymbol{C} &= \boldsymbol{A} : \mathbb{B}

        C_{kl} &= A_{ij} : \mathbb{B}_{ijkl}

    and by Eq. :eq:`math-ddot42` for a fourth-order and a second-order tensor.

    ..  math::
        :label: math-ddot42

        \boldsymbol{C} &= \mathbb{A} : \boldsymbol{A}

        C_{ij} &= \mathbb{A}_{ijkl} : B_{kl}


    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> A = fem.math.transpose(np.arange(9, dtype=float).reshape(1, 3, 3).T)
    >>> B = A + 10

    >>> fem.math.ddot(A, B)[..., 0]
    array(564.)

    >>> fem.math.ddot(fem.math.dya(A, A), B, mode=(4, 2))[..., 0]
    array([[   0.,  564., 1128.],
           [1692., 2256., 2820.],
           [3384., 3948., 4512.]])

    """

    if parallel:
        einsum = einsumt
    else:
        einsum = np.einsum

    if mode == (2, 2):
        return einsum("ij...,ij...->...", A, B, **kwargs)
    elif mode == (2, 4):
        return einsum("ij...,ijkl...->kl...", A, B, **kwargs)
    elif mode == (4, 2):
        return einsum("ijkl...,kl...->ij...", A, B, **kwargs)
    elif mode == (2, 3):
        return einsum("ij...,ijk...->k...", A, B, **kwargs)
    elif mode == (3, 2):
        return einsum("ijk...,jk...->i...", A, B, **kwargs)
    elif mode == (4, 4):
        return einsum("ijkl...,klmn...->ijmn...", A, B, **kwargs)
    else:
        raise TypeError("Unknown shape of A and B.")


def tovoigt(A, strain=False):
    r"""Return a three-dimensional second-order tensor in reduced symmetric (Voigt-
    notation) vector/matrix storage.

    Parameters
    ----------
    A : ndarray of shape (2, 2, ...) or (3, 3, ...)
        Array with three-dimensional second-order tensors.
    strain : bool, optional
        A flag to double the off-diagonal (shear) values for strain-like tensors.
        Default is False.

    Returns
    -------
    ndarray of shape (3, ...) or (6, ...)
        A three-dimensional second-order tensor in reduced symmetric (Voigt) vector/
        matrix storage.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.
    
    For a symmetric three-dimensional second-order tensor :math:`C_{ij} = C_{ji}`, the
    upper triangle entries are inserted into a 6x1 vector, starting from the main
    diagonal, followed by the consecutive next upper diagonals.

    ..  math::

        \boldsymbol{C} = \begin{bmatrix}
            C_{11} & C_{12} & C_{13} \\
            C_{12} & C_{22} & C_{23} \\
            C_{13} & C_{23} & C_{33}
        \end{bmatrix} \qquad \longrightarrow \boldsymbol{C} = \begin{bmatrix}
            C_{11} \\ C_{22} \\ C_{33} \\ C_{12} \\ C_{23} \\ C_{13}
        \end{bmatrix}

    Examples
    --------
    >>> import felupe as fem
    >>> import numpy as np
    >>>
    >>> C = np.array([1.0, 1.3, 1.5, 1.3, 1.1, 1.4, 1.5, 1.4, 1.2]).reshape(3, 3, 1)
    >>> C[..., 0]
    array([[1. , 1.3, 1.5],
           [1.3, 1.1, 1.4],
           [1.5, 1.4, 1.2]])
    
    >>> fem.math.tovoigt(C)[..., 0]
    array([1. , 1.1, 1.2, 1.3, 1.4, 1.5])

    """
    dim = A.shape[:2]
    if dim == (1, 1):
        B = np.zeros((1, *A.shape[2:]))
        ij = [(0, 0)]
    elif dim == (2, 2):
        B = np.zeros((3, *A.shape[2:]))
        ij = [(0, 0), (1, 1), (0, 1)]
    elif dim == (3, 3):
        B = np.zeros((6, *A.shape[2:]))
        ij = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    else:
        raise TypeError("Input shape must be (2, 2, ...) or (3, 3, ...).")
    for a, (i, j) in enumerate(ij):
        B[a] = A[i, j]
    if strain:
        B[dim[0] :] *= 2
    return B


def reshape(A, shape, trailing_axes=2):
    return A.reshape(np.append(shape, A.shape[-trailing_axes:]))


def ravel(A, trailing_axes=2):
    ij, shape = np.split(A.shape, [-trailing_axes])
    return reshape(A, shape=np.prod(ij))


def equivalent_von_mises(A):
    r"""Return the Equivalent von Mises values of symmetric second order-tensors.

    Parameters
    ----------
    A : ndarray of shape (2, 2, ...) or (3, 3, ...)
        Symmetric second-order tensors for which the equivalent von Mises values will be
        computed.

    Returns
    -------
    ndarray of shape (...)
        The equivalent von Mises values.

    Notes
    -----
    The first two axes are the tensor dimensions and all remaining trailing axes are
    treated as batch dimensions.

    The equivalent von Mises value of a three-dimensional symmetric second order-tensor
    is given in Eq. :eq:`math-vonmises`.

    ..  math::
        :label: math-vonmises

        \boldsymbol{A}_{v} = \sqrt{
            \frac{3}{2} \text{dev}(\boldsymbol{A}) : \text{dev}(\boldsymbol{A})
        }

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> stress = np.array([1, 1.1, 1.2, 1.1, 1.3, 1.4, 1.2, 1.4, 1.5]).reshape(3, 3, 1)
    >>> fem.math.equivalent_von_mises(stress)
    array([3.74432905])

    >>> stress = np.diag([3, 1, 1]).reshape(3, 3, 1)
    >>> fem.math.equivalent_von_mises(stress)
    array([2.])
    """

    pad_width = len(A.shape) * [(0, 0)]
    pad_width[0] = (0, 3 - A.shape[0])
    pad_width[1] = (0, 3 - A.shape[1])

    A = np.pad(A, pad_width)

    devA = dev(A)
    return np.sqrt(3 / 2 * ddot(devA, devA))

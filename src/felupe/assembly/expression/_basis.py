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

try:
    from einsumt import einsumt
except ModuleNotFoundError:
    from numpy import einsum as einsumt


class BasisArray(np.ndarray):
    """Add the grad- and hess-attributes to an existing array [1]_, [2]_.

    Parameters
    ----------
    input_array : array_like
        The input array.
    grad : array_like or None, optional
        The array for the grad-attribute (default is None).
    hess : array_like or None, optional
        The array for the hess-attribute (default is None).

    Examples
    --------
    ..  pyvista-plot::

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> x = fem.assembly.expression.BasisArray(
        >>>     np.ones(3), grad=np.zeros((3, 3)), hess=np.zeros((3, 3, 3))
        >>> )
        >>> x
        BasisArray([1., 1., 1.])

        >>> x.grad
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

        >>> x.hess
        array([[[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]],
        <BLANKLINE>
               [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]],
        <BLANKLINE>
               [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]])

    References
    ----------
    ..  [1] https://numpy.org/doc/stable/user/basics.subclassing.html

    ..  [2] https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """

    def __new__(cls, input_array, grad=None, hess=None):
        obj = np.asarray(input_array).view(cls)
        obj.grad = grad
        obj.hess = hess
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grad = getattr(obj, "grad", None)
        self.hess = getattr(obj, "hess", None)


class BasisField:
    r"""Basis and its gradient for a field.

    Parameters
    ----------
    field : Field
        A field on which the basis is created.
    parallel : bool, optional (default is False)
        Flag to activate parallel (threaded) basis evaluation.

    Attributes
    ----------
    basis : ndarray
        The evaluated basis functions at quadrature points.
    grad : ndarray
        The evaluated gradient of the basis (if provided by the region).
    hess : ndarray
        The evaluated hessian of the basis (if provided by the region).

    Notes
    -----
    *Basis* refers to the trial and test field, either values or gradients evaluated at
    quadrature points. The first two indices of a basis are used for looping over the
    element shape functions ``a`` and its components ``i``. The third index represents
    the vector component ``j`` of the field. The two trailing axes ``(q, c)`` contain
    the evaluated element shape functions at quadrature points ``q`` per cell ``c``.

    ..  math::

        \varphi_{ai~j~qc} = \delta_{ij} \left( h_a \right)_{qc}

    For gradients, the fourth index is used for the vector component of the
    partial derivative ``k``.

    ..  math::

        \text{grad}(\varphi)_{ai~jK~qc} = \delta_{ij}
            \left( \frac{\partial h_a}{\partial X_K} \right)_{qc}

    Examples
    --------
    ..  pyvista-plot::

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle()
        >>> region = fem.RegionQuad(mesh)
        >>> displacement = fem.Field(region, dim=2)
        >>>
        >>> bf = fem.assembly.expression.BasisField(displacement)
        >>> bf.basis.shape
        (4, 2, 2, 4, 1)

        >>> bf.basis.grad.shape
        (4, 2, 2, 2, 4, 1)

        >>> bf.basis.hess.shape
        (4, 2, 2, 2, 2, 4, 1)

    See Also
    --------
    felupe.assembly.expression.Basis : Bases and their gradients for the fields of a
        field container.

    """

    def __init__(self, field, parallel=False):
        self.field = field

        einsum = einsumt if parallel else np.einsum

        basis = einsum(
            "ij,aqc->aijqc",
            np.eye(self.field.dim),
            self.field.region.h,
        )

        if self.field.region.evaluate_gradient:
            grad = einsum(
                "ij,akqc->aijkqc", np.eye(self.field.dim), self.field.region.dhdX
            )

        else:
            grad = np.full(basis.shape[:2], None)

        if self.field.region.evaluate_hessian:
            hess = einsum(
                "ij,aklqc->aijklqc", np.eye(self.field.dim), self.field.region.d2hdXdX
            )

        else:
            hess = np.full(basis.shape[:2], None)

        self.basis = BasisArray(basis, grad=grad, hess=hess)


class Basis:
    r"""Bases and their gradients for the fields of a field container.

    Parameters
    ----------
    field : FieldContainer
        A field container on which the basis should be created.

    Attributes
    ----------
    basis : ndarray
        The list of bases.

    Examples
    --------
    ..  pyvista-plot::

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle()
        >>> region = fem.RegionQuad(mesh)
        >>> displacement = fem.Field(region, dim=2)
        >>> field = fem.FieldContainer([displacement])
        >>>
        >>> bases = fem.assembly.expression.Basis(field)
        >>> len(bases[:])
        >>> 1

        >>> bases[0].basis.shape
        (4, 2, 2, 4, 1)

        >>> bases[0].basis.grad.shape
        (4, 2, 2, 2, 4, 1)

        >>> bases[0].basis.hess.shape
        (4, 2, 2, 2, 2, 4, 1)

    See Also
    --------
    felupe.assembly.expression.BasisField : Basis and its gradient for a field.

    """

    def __init__(self, field, parallel=False):
        self.field = field
        self.basis = [BasisField(f, parallel=parallel) for f in self.field]

    def __getitem__(self, idx):
        "Slice-based access to underlying bases."

        return self.basis[idx]

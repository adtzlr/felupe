# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

try:
    from einsumt import einsumt
except:
    from numpy import einsum as einsumt


class Basis:
    r"""A basis and its gradient built on top of a scalar- or vector-valued
    field. *Basis* refers to the trial and test field, either values or
    gradients evaluated at quadrature points. The first two indices of a basis
    are used for looping over the element shape functions ``a`` and its
    components ``i``. The third index represents the vector component ``j`` of
    the field. The two trailing axes ``(p, c)`` contain the evaluated element
    shape functions at quadrature points ``p`` per cell ``c``.

    ..  math::

        \varphi_{aijpc} = \delta_{ij} \left( h_a \right)_{pc}


    For gradients, the fourth index is used for the vector component of the
    partial derivative ``k``.

    ..  math::

        \text{grad}(\varphi)_{aijkpc} = \delta_{ij}
            \left( \frac{\partial h_a}{\partial X_K} \right)_{pc}

    Parameters
    ----------
    field : Field
        A field on which the basis should be created.
    parallel : bool, optional (default is False)
        Flag to activate parallel (threaded) basis evaluation.

    Attributes
    ----------
    basis : ndarray
        The evaluated basis functions at quadrature points.
    grad : ndarray
        The evaluated gradient of the basis (if provided by the region).

    """

    def __init__(self, field, parallel=False):

        self.field = field

        einsum = einsumt if parallel else np.einsum

        self.basis = einsum(
            "ij,apc->aijpc",
            np.eye(self.field.dim),
            self.field.region.h,
        )

        if hasattr(self.field.region, "dhdX"):
            self.grad = einsum(
                "ij,akpc->aijkpc", np.eye(self.field.dim), self.field.region.dhdX
            )

        else:
            self.grad = None


class BasisMixed:
    r"""A basis and its gradient built on top of a scalar- or vector-valued
    field container. *Basis* refers to the trial and test field, either values or
    gradients evaluated at quadrature points. The first two indices of a basis
    are used for looping over the element shape functions ``a`` and its
    components ``i``. The third index represents the vector component ``j`` of
    the field. The two trailing axes ``(p, c)`` contain the evaluated element
    shape functions at quadrature points ``p`` per cell ``c``.

    ..  math::

        \varphi_{aijpc} = \delta_{ij} \left( h_a \right)_{pc}


    For gradients, the fourth index is used for the vector component of the
    partial derivative ``k``.

    ..  math::

        \text{grad}(\varphi)_{aijkpc} = \delta_{ij}
            \left( \frac{\partial h_a}{\partial X_K} \right)_{pc}

    Parameters
    ----------
    field : FieldContainer
        A field container on which the basis should be created.

    Attributes
    ----------
    basis : ndarray
        The evaluated basis functions at quadrature points.
    grad : ndarray
        The evaluated gradient of the basis (if provided by the region).

    """

    def __init__(self, field, parallel=False):

        self.field = field
        self.basis = [Basis(f, parallel=parallel) for f in self.field]

    def __getitem__(self, idx):
        "Slice-based access to underlying bases."

        return self.basis[idx]

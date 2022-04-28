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
from copy import deepcopy
from ..math import identity, sym as symmetric
from ._indices import Indices


class Field:
    r"""A Field on points of a `region` with dimension `dim`
    and initial point `values`. A slice of this field directly
    accesses the field values as 1d-array.

    The interpolation method returns the field values evaluated at the
    numeric integration points ``p`` of all cells ``c`` in the region.

    ..  math::

        u^i_{(pc)} = \hat{u}_a^i h_{a(pc)}

    The gradient method returns the gradient of the field values w.r.t. the
    undeformed mesh point coordinates, evaluated at the integration points of
    all cells in the region.

    ..  math::

        \left( \frac{\partial u^i}{\partial X^J} \right)_{(pc)} =
        \hat{u}^i_{a(pc)}
        \left( \frac{\partial h_a}{\partial X^J} \right)_{(pc)}

    Arguments
    ---------
    region : Region
        The region on which the field will be created.
    dim : int (default is 1)
        The dimension of the field.
    values : float (default is 0.0) or array
        A single value for all components of the field or an array of
        shape (region.mesh.npoints, dim)`.
    kwargs : dict, optional
        Optional keyword arguments of the field.

    """

    def __init__(self, region, dim=1, values=0, **kwargs):

        self.region = region
        self.dim = dim
        self.shape = self.region.quadrature.npoints, self.region.mesh.ncells

        # set optional user-defined attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # init values
        if isinstance(values, np.ndarray):
            if len(values) == region.mesh.npoints:
                self.values = values
            else:
                raise ValueError("Wrong shape of values.")

        else:  # scalar value
            self.values = np.ones((region.mesh.npoints, dim)) * values

        eai, ai = self._indices_per_cell(self.region.mesh.cells, dim)
        self.indices = Indices(eai, ai, region, dim)

    def _indices_per_cell(self, cells, dim):
        "Calculate pre-defined indices for sparse matrices."

        # index of cell "e", point "a" and component "i"
        eai = (
            dim * np.repeat(cells, dim) + np.tile(np.arange(dim), cells.size)
        ).reshape(*cells.shape, dim)
        # store indices as (rows, cols) (note: sparse-matrices are always 2d)
        ai = (eai.ravel(), np.zeros_like(eai.ravel()))

        return eai, ai

    def grad(self, sym=False):
        """Gradient as partial derivative of field values at points w.r.t.
        undeformed coordinates, evaluated at the integration points of
        all cells in the region. Optionally, the symmetric part of
        the gradient is evaluated.

        Arguments
        ---------
        sym : bool, optional (default is False)
            Calculate the symmetric part of the gradient.

        Returns
        -------
        array
            Gradient as partial derivative of field values at points w.r.t.
            undeformed coordinates, evaluated at the integration points of
            all cells in the region.
        """

        # gradient dudX_IJpe as partial derivative of field values at points "aI"
        # w.r.t. undeformed coordinates "J" evaluated at quadrature point "p"
        # for each cell "c"
        g = np.einsum(
            "ca...,aJpc->...Jpc",
            self.values[self.region.mesh.cells],
            self.region.dhdX,
        )

        if sym:
            return symmetric(g)
        else:
            return g

    def interpolate(self):
        """Interpolate field values at points and evaluate them at the
        integration points of all cells in the region."""

        # interpolated field values "aI"
        # evaluated at quadrature point "p"
        # for cell "c"
        return np.einsum(
            "ca...,apc->...pc", self.values[self.region.mesh.cells], self.region.h
        )

    def extract(self, grad=True, sym=False, add_identity=True):
        """Generalized extraction method which evaluates either the gradient
        or the field values at the integration points of all cells
        in the region. Optionally, the symmetric part of the gradient is
        evaluated and/or the identity matrix is added to the gradient.

        Arguments
        ---------
        grad : bool, optional (default is True)
            Flag for gradient evaluation.
        sym : bool, optional (default is False)
            Flag for symmetric part if the gradient is evaluated.
        add_identity : bool, optional (default is True)
            Flag for the addition of the identity matrix
            if the gradient is evaluated.

        Returns
        -------
        array
            (Symmetric) gradient or interpolated field values evaluated at
            the integration points of each cell in the region.
        """

        if grad:
            gr = self.grad()

            if sym:
                gr = symmetric(gr)

            if add_identity:
                gr = identity(gr) + gr

            return gr
        else:
            return self.interpolate()

    def copy(self):
        "Return a copy of the field."
        return deepcopy(self)

    def fill(self, a):
        "Fill all field values with a scalar value."
        self.values.fill(a)

    def __add__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values += newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values += newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __sub__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values -= newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values -= newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __mul__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values *= newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values *= newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __truediv__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            field = deepcopy(self)
            field.values /= newvalues.reshape(-1, field.dim)
            return field

        elif isinstance(newvalues, Field):
            field = deepcopy(self)
            field.values /= newvalues.values
            return field

        else:
            raise TypeError("Unknown type.")

    def __iadd__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            self.values += newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values += newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __isub__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            self.values -= newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values -= newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __imul__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            self.values *= newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values *= newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __itruediv__(self, newvalues):

        if isinstance(newvalues, np.ndarray):
            self.values /= newvalues.reshape(-1, self.dim)
            return self

        elif isinstance(newvalues, Field):
            self.values /= newvalues.values
            return self

        else:
            raise TypeError("Unknown type.")

    def __getitem__(self, dof):
        """Slice-based access to a 1d-representation of field values by
        a list of degrees of freedom `dof`."""

        return self.values.ravel()[dof]

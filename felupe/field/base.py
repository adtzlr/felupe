# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:29:15 2021

@author: adutz
"""
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
from .indices import Indices


class Field:
    "n-dimensional continous field in region."

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

        eai, ai = self.indices_per_cell(self.region.mesh.cells, dim)
        self.indices = Indices(eai, ai, region, dim)

    def indices_per_cell(self, cells, dim):
        "Pre-defined indices for sparse matrices."

        # index of cell "e", point "a" and component "i"
        eai = np.stack(
            [dim * np.tile(cell, (dim, 1)).T + np.arange(dim) for cell in cells]
        )
        # store indices as (rows, cols) (note: sparse-matrices are always 2d)
        ai = (eai.ravel(), np.zeros_like(eai.ravel()))

        return eai, ai

    def grad(self, sym=False):
        "gradient dudX_IJpe"
        # gradient as partial derivative of field values at points "aI"
        # w.r.t. undeformed coordinate "J" evaluated at quadrature point "p"
        # for cell "e"
        g = np.einsum(
            "ea...,aJpe->...Jpe", self.values[self.region.mesh.cells], self.region.dhdX,
        )
        if sym:
            return symmetric(g)
        else:
            return g

    def interpolate(self):
        "interpolated values u_Ipe"
        # interpolated field values "aI"
        # evaluated at quadrature point "p"
        # for cell "e"
        return np.einsum(
            "ea...,ap->...pe", self.values[self.region.mesh.cells], self.region.h
        )

    def extract(self, grad=True, sym=False, add_identity=True):
        "Extract gradient or interpolated field values at quadrature points."

        if grad:
            gr = self.grad()

            if sym:
                gr = sym(gr)

            if add_identity:
                gr = identity(gr) + gr

            return gr
        else:
            return self.interpolate()

    def copy(self):
        return deepcopy(self)

    def fill(self, a):
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
        "Slice-based access to flattened values by list of dof's."

        return self.values.ravel()[dof]

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

from copy import copy, deepcopy

from .math import identity


def extract(fields, fieldgrad=(True, False), add_identity=False):
    grads = np.pad(fieldgrad, (0, len(fields)))
    list_of_fields = []
    for grad, field in zip(grads, fields):
        if grad:
            f = field.grad()
            if add_identity:
                f += identity(f)
        else:
            f = field.interpolate()
        list_of_fields.append(f)
    return list_of_fields


class Indices:
    def __init__(self, eai, ai, region, dim):
        self.eai = eai
        self.ai = ai
        self.dof = np.arange(region.nnodes * dim).reshape(-1, dim)
        self.shape = (region.nnodes * dim, 1)


class Field:
    "n-dimensional field in region."

    def __init__(self, region, dim=1, values=0):
        self.region = region
        self.dim = dim

        # init values
        self.values = np.ones((region.nnodes, dim)) * values

        eai, ai = self.indices_per_element(self.region.connectivity, dim)
        self.indices = Indices(eai, ai, region, dim)

    def indices_per_element(self, connectivity, dim):
        "Pre-defined indices for sparse matrices."

        # index of element "e", node "a" and nodal-value component "i"
        eai = np.stack(
            [dim * np.tile(conn, (dim, 1)).T + np.arange(dim) for conn in connectivity]
        )
        # store indices as (rows, cols) (note: sparse-matrices are always 2d)
        ai = (eai.ravel(), np.zeros_like(eai.ravel()))

        return eai, ai

    def grad(self):
        "gradient dudX_IJpe"
        # gradient as partial derivative of nodal field values "aI"
        # w.r.t. undeformed coordiante "J" evaluated at quadrature point "p"
        # for element "e"
        return np.einsum(
            "ea...,aJpe->...Jpe",
            self.values[self.region.connectivity],
            self.region.dhdX,
        )

    def interpolate(self):
        "interpolated values u_Ipe"
        # interpolated field values "aI"
        # evaluated at quadrature point "p"
        # for element "e"
        return np.einsum(
            "ea...,ap->...pe", self.values[self.region.connectivity], self.region.h
        )

    def copy(self):
        out = copy(self)
        return out

    def full(self, a):
        self.values = np.full(self.values.shape, a, dtype=np.float)

    def fill(self, a):
        self.full(a)

    def __add__(self, newvalues):
        self.__iadd__(newvalues)

    def __sub__(self, newvalues):
        self.__isub__(newvalues)

    def __mul__(self, newvalues):
        self.__imul__(newvalues)

    def __truediv__(self, newvalues):
        self.__itruediv__(newvalues)

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

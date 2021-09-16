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
from .tensor import identity, transpose


def defgrad(field):
    return field.extract(grad=True, sym=False, add_identity=True)


def strain(field):
    return field.grad(sym=True)


def extract(field, grad=True, sym=False, add_identity=True):
    "Extract gradient or interpolated field values at quadrature points."
    return field.extract(grad=grad, sym=sym, add_identity=add_identity)


def values(field):
    "Return values of a field or a tuple of fields."
    
    if "mixed" in str(type(field)).lower():
        field = field.fields

    if isinstance(field, tuple) or isinstance(field, list):
        return np.concatenate([f.values.ravel() for f in field])

    else:
        return field.values.ravel()


def norm(array):
    "Calculate the norm of an array."
    return np.linalg.norm(array)


def norms(arrays):
    "Calculate norms of a list of arrays."
    return np.array([np.linalg.norm(arr) for arr in arrays])


def interpolate(field):
    "Interpolate method of field A."
    return field.interpolate()


def grad(field, sym=False):
    "Gradient method of field A."
    return field.grad(sym=sym)


def sym(A):
    "Symmetric part of matrix A."
    return (A + transpose(A)) / 2


def laplace(field):
    n = field.dim
    m = field.region.mesh.ndim
    p = field.region.quadrature.npoints
    e = field.region.mesh.ncells

    return identity(field.grad()).reshape(n, m, n, m, p, e)

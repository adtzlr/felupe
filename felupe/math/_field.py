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


def defgrad(field):
    return field.extract(grad=True, sym=False, add_identity=True)


def strain(field):
    return field.grad(sym=True)


def extract(field, grad=True, sym=False, add_identity=True):
    "Extract gradient or interpolated field values at quadrature points."
    return field.extract(grad=grad, sym=sym, add_identity=add_identity)


def values(field):
    "Return values of a field or a tuple of fields."

    return np.concatenate([f.values.ravel() for f in field.fields])


def norm(array, axis=None):
    "Calculate the norm of an array or the norms of a list of arrays."
    if isinstance(array, list):
        return np.array([np.linalg.norm(arr, axis=axis) for arr in array])
    else:
        return np.linalg.norm(array, axis=axis)


def interpolate(field):
    "Interpolate method of field A."
    return field.interpolate()


def grad(field, sym=False):
    "Gradient method of field A."
    return field.grad(sym=sym)

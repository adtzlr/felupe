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
from scipy.interpolate import interp1d
from scipy.sparse import issparse

from ..math import cross


def force(field, forces, boundary):
    "Evaluate the force vector sum on points of a boundary."

    if issparse(forces):
        forces = forces.toarray()

    forces_first_field = np.split(forces, field.offsets)[0]
    dim = field[0].dim

    return ((forces_first_field.reshape(-1, dim))[boundary.points]).sum(axis=0)


def moment(field, forces, boundary, centerpoint=np.zeros(3)):
    "Evaluate the moment vector sum on points of a boundary at a given center point."

    if issparse(forces):
        forces = forces.toarray()

    dim = field[0].dim
    centerpoint = np.asarray(centerpoint).reshape(1, -1)[:, :dim]

    displacements = field[0].values
    force = (np.split(forces, field.offsets)[0]).reshape(-1, dim)

    moments = cross(
        (field.region.mesh.points + displacements - centerpoint)[boundary.points].T,
        force[boundary.points].T,
    ).T

    return moments.sum(axis=0)


def curve(x, y, num=50):
    "Interpolate a curve from given (x, y) data."

    kind = [None, "linear", "quadratic", "cubic"][min(len(y), 4) - 1]

    f = interp1d(x[: len(y)], y, kind=kind)

    xt = x[: len(y)]
    xx = np.linspace(xt[0], xt[-1], num=num)

    return np.array([xt, y]), np.array([xx, f(xx)])

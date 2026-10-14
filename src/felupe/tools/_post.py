# SPDX-License-Identifier: GPL-3.0-or-later

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

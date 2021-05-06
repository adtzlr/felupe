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
from types import SimpleNamespace


def get_dof0(field, bounds):
    "Extract prescribed degrees of freedom."
    mesh = field.region.mesh
    dim = field.dim
    fixmissing = dim * np.tile(mesh.nodes_without_elements, (dim, 1)).T + np.arange(dim)
    dof0_bounds = np.concatenate([b.dof for b in bounds.values()])
    return np.unique(np.append(fixmissing.ravel(), dof0_bounds))


def get_dof1(field, bounds, dof0=None):
    "Extract active (non-prescribed) degrees of freedom."
    dof = field.indices.dof
    if dof0 is None:
        dof0 = get_dof0(field, bounds)
    mask = np.ones_like(dof.ravel(), dtype=bool)
    mask[dof0] = False
    return dof.ravel()[mask]


find = SimpleNamespace()
find.dof0 = get_dof0
find.dof1 = get_dof1


def partition(field, bounds):
    "Partition dof-list to prescribed and active parts."

    dof0 = get_dof0(field, bounds)
    dof1 = get_dof1(field, bounds, dof0=dof0)

    return dof0, dof1


def extend(fields, dof0, dof1):

    fieldsizes = [f.indices.dof.size for f in fields]
    offsets = np.cumsum(fieldsizes)

    dof0_xt = dof0.copy()
    dof1_xt = dof1.copy()

    for field, offset, fieldsize in zip(fields[1:], offsets[:-1], fieldsizes[1:]):

        mesh = field.region.mesh
        dim = field.dim
        dof0_add = dim * np.tile(mesh.nodes_without_elements, (dim, 1)).T + np.arange(
            dim
        )
        dof1_add = dim * np.tile(mesh.nodes_with_elements, (dim, 1)).T + np.arange(dim)

        # dof1_xt = np.append(dof1_xt, offset + np.arange(fieldsize))

        dof0_xt = np.append(dof0_xt, offset + dof0_add.ravel())
        dof1_xt = np.append(dof1_xt, offset + dof1_add.ravel())

    return dof0_xt, dof1_xt, offsets[:-1]


def apply(v, bounds, dof0=None):
    """Apply prescribed values for a list of boundaries
    and return all (default) or only the prescribed components
    of the input array 'v' based on the keyword 'dof0'."""

    u = v.values.copy()

    for b in bounds.values():
        u.ravel()[b.dof] = b.value

    if dof0 is None:
        return u
    else:
        u0ext = u.ravel()[dof0[dof0 < u.size]]
        u0ext_padded = np.pad(u0ext, (0, len(dof0) - len(u0ext)))
        return u0ext_padded


def symmetry(field, axes=(True, True, True), x=0, y=0, z=0):

    mesh = field.region.mesh

    axes = np.array(axes).astype(bool)[: mesh.ndim]

    fx = lambda v: np.isclose(v, x)
    fy = lambda v: np.isclose(v, y)
    fz = lambda v: np.isclose(v, z)

    bounds = []
    skipax = ~np.eye(3).astype(bool)
    kwarglist = [
        {"fx": fx, "skip": skipax[0][: mesh.ndim]},
        {"fy": fy, "skip": skipax[1][: mesh.ndim]},
        {"fz": fz, "skip": skipax[2][: mesh.ndim]},
    ]

    bounds = {}
    ax = ["x", "y", "z"]
    for a, (enforce_sym, kwargs) in enumerate(zip(axes, kwarglist[: mesh.ndim])):
        if enforce_sym:
            bounds["sym" + ax[a]] = Boundary(field, **kwargs)

    return bounds


class Boundary:
    def __init__(
        self,
        field,
        name="default",
        fx=lambda v: v == np.nan,
        fy=lambda v: v == np.nan,
        fz=lambda v: v == np.nan,
        value=0,
        skip=(False, False, False),
    ):

        mesh = field.region.mesh
        dof = field.indices.dof

        self.ndim = mesh.ndim
        self.name = name
        self.value = value
        self.skip = np.array(skip).astype(int)[: self.ndim]
        self.fun = [fx, fy, fz][: self.ndim]

        # apply functions on the nodes per coordinate
        # fx(x), fy(y), fz(z) and create a mask for each coordinate
        mask = [f(x) for f, x in zip(self.fun, mesh.nodes.T)]

        # combine the masks with logical or
        tmp = np.logical_or(mask[0], mask[1])
        if self.ndim == 3:
            mask = np.logical_or(tmp, mask[2])
        else:
            mask = tmp

        # tile the mask
        self.mask = np.tile(mask.reshape(-1, 1), self.ndim)

        # check if some axes should be skipped
        if True not in skip:
            pass
        else:
            # exclude mask from axes which should be skipped
            self.mask[:, np.where(self.skip)[0]] = False

        self.dof = dof[self.mask]

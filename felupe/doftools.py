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

def get_dof0(dof, bounds):
    "Extract prescribed degrees of freedom."
    return np.unique(np.concatenate([b.dof for b in bounds]))


def get_dof1(dof, bounds, dof0=None):
    "Extract active (non-prescribed) degrees of freedom."
    if dof0 is None:
        dof0 = get_dof0(dof, bounds)
    mask = np.ones_like(dof.ravel(), dtype=bool)
    mask[dof0] = False
    return dof.ravel()[mask]


find = SimpleNamespace()
find.dof0 = get_dof0
find.dof1 = get_dof1


def partition(dof, bounds):
    "Partition dof-list to prescribed and active parts."
    dof0 = get_dof0(dof, bounds)
    dof1 = get_dof1(dof, bounds, dof0=dof0)
    return dof0, dof1


def apply(v, dof, bounds, dof0=None):
    """Apply prescribed values for a list of boundaries
    and return all (default) or only the prescribed components
    of the input array 'v' based on the keyword 'dof0'."""

    u = v.copy()

    for b in bounds:
        u.ravel()[b.dof] = b.value

    if dof0 is None:
        return u
    else:
        return u.ravel()[dof0]


class Boundary:
    def __init__(
        self,
        dof,
        mesh,
        name="default",
        fx=lambda x: x == np.nan,
        fy=lambda y: y == np.nan,
        fz=lambda z: z == np.nan,
        value=0,
        skip=(False, False, False),
    ):

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

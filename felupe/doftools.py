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


def dof0(dof, bounds):
    return np.unique(np.concatenate([b.dof for b in bounds]))


def dof1(dof, bounds):
    keep = np.ones_like(dof.ravel(), dtype=bool)
    dismiss = dof0(dof, bounds)
    keep[dismiss] = False
    return dof.ravel()[keep]


def partition(dof, bounds):
    D = dof0(dof, bounds)
    I = dof1(dof, bounds)
    return D, I


def apply(v, dof, bounds, D=None):
    u = v.copy()
    for b in bounds:
        u.ravel()[b.dof] = b.value
    if D is None:
        return u
    else:
        return u.ravel()[D]


class Boundary:
    def __init__(
        self,
        dof,
        mesh,
        name,
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

        mask = [f(x) for f, x in zip(self.fun, mesh.nodes.T)]

        tmp = np.logical_or(mask[0], mask[1])
        if self.ndim == 3:
            mask = np.logical_or(tmp, mask[2])
        else:
            mask = tmp

        self.mask = np.tile(mask.reshape(-1, 1), self.ndim)
        if True not in skip:
            pass
        else:
            self.mask = np.tile(mask.reshape(-1, 1), self.ndim)
            for i in np.where(self.skip == 1):
                self.mask[:, i] = False

        self.dof = dof[self.mask]

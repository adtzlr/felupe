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
import pytest
import numpy as np
import felupe as fe


def test_helpers():
    mesh = fe.mesh.Cube()
    dof = np.arange(mesh.ndof).reshape(mesh.nodes.shape)

    u = np.zeros_like(mesh.nodes)

    f0 = lambda x: x == 0
    f1 = lambda x: x == 1
    v = 13.2
    left = fe.doftools.Boundary(dof, mesh, fx=f0, skip=(0, 0, 1), value=v)
    right = fe.doftools.Boundary(dof, mesh, fx=f1, skip=(0, 1, 1))

    bounds = [left, right]

    dof0, dof1 = fe.doftools.partition(dof, bounds)

    uext = fe.doftools.apply(u, dof, bounds, dof0=None)
    u0ext = fe.doftools.apply(u, dof, bounds, dof0=dof0)

    dof01 = np.sort(np.append(dof0, dof1))
    if not np.all(np.equal(dof.ravel(), dof01)):
        raise ValueError("Partitioning of DOF failed.")

    if not np.allclose(uext.ravel()[dof0], u0ext):
        raise ValueError("Application of external values failed.")

    if not np.allclose(uext.ravel()[left.dof], v):
        raise ValueError("Application of external values failed.")

    return mesh, dof, u, uext, left, right, dof0, dof1


if __name__ == "__main__":
    mesh, dof, u, uext, left, right, dof0, dof1 = test_helpers()

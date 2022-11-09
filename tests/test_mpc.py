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


def test_mpc():

    mesh = fe.Cube(n=3)
    mesh.points = np.vstack((mesh.points, [2, 0, 0]))
    mesh.update(mesh.cells)

    region = fe.RegionHexahedron(mesh)

    u = fe.FieldContainer([fe.Field(region, dim=3)])
    F = u.extract()

    umat = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fe.Boundary(u[0], fx=f0)
    boundaries["right"] = fe.Boundary(u[0], fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(u[0], fx=f2, skip=(0, 1, 1), value=0.5)

    mpc = fe.Boundary(u[0], fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fe.MultiPointConstraint(field=u, points=mpc, centerpoint=cpoint)

    for f in [None, u]:
        K_RBE2 = RBE2.assemble.matrix(u)
        r_RBE2 = RBE2.assemble.vector(u)

        linearform = fe.IntegralForm(umat.gradient(F)[:-1], u, region.dV)
        r = linearform.assemble() + r_RBE2

        bilinearform = fe.IntegralForm(umat.hessian(F), u, region.dV, u)
        K = bilinearform.assemble() + K_RBE2

        assert r.shape == (84, 1)
        assert K.shape == (84, 84)


def pre_mpc_mixed(point, values):

    mesh = fe.mesh.Cube(n=3)
    mesh.points = np.vstack((mesh.points, point))
    mesh.update(mesh.cells)

    region = fe.RegionHexahedron(mesh)
    dV = region.dV

    fields = fe.FieldsMixed(region, n=3)
    fields[0].values[-1] = values
    F, p, J = fields.extract()

    nh = fe.NeoHooke(mu=1.0, bulk=2.0)
    umat = fe.ThreeFieldVariation(nh)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fe.Boundary(fields[0], fx=f0)
    boundaries["right"] = fe.Boundary(fields[0], fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(fields[0], fx=f2, skip=(0, 1, 1), value=0.5)

    mpc = fe.Boundary(fields[0], fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fe.MultiPointConstraint(fields, points=mpc, centerpoint=cpoint)
    CONT = fe.MultiPointContact(fields, points=mpc, centerpoint=cpoint)

    for f in [None, fields]:
        K_RBE2 = RBE2.assemble.matrix(f)
        r_RBE2 = RBE2.assemble.vector(f)

        K_CONT = CONT.assemble.matrix(f)
        r_CONT = CONT.assemble.vector(f)

        assert K_RBE2.shape == K_CONT.shape
        assert r_RBE2.shape == r_CONT.shape

    linearform = fe.IntegralForm(umat.gradient([F, p, J])[:-1], fields, dV)
    r = linearform.assemble()

    r_RBE2.resize(*r.shape)
    r = r + r_RBE2

    bilinearform = fe.IntegralForm(umat.hessian([F, p, J]), fields, dV, fields)
    K = bilinearform.assemble()

    K_RBE2.resize(*K.shape)
    K = K + K_RBE2


def test_mpc_mixed():
    pre_mpc_mixed(point=[2, 0, 0], values=[0, 0, 0])
    pre_mpc_mixed(point=[2, 0, 0], values=[-5, 0, 0])


if __name__ == "__main__":
    test_mpc()
    test_mpc_mixed()

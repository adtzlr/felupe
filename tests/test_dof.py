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
from copy import deepcopy


def pre1d():
    m = fe.mesh.Line()
    e = fe.element.Line()
    q = fe.quadrature.GaussLegendre(1, 1)
    r = fe.Region(m, e, q)
    u = fe.Field(r)
    return u


def pre2d():
    m = fe.mesh.Rectangle()
    e = fe.element.Quad()
    q = fe.quadrature.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=2)
    return u


def pre3d():
    m = fe.mesh.Cube()
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=3)
    return u


def test_boundary():

    u = pre3d()
    bounds = {"boundary-label": fe.Boundary(u)}

    v = fe.dof.apply(u, bounds, dof0=None)
    assert np.allclose(u.values, v)

    mask = np.ones(u.region.mesh.npoints, dtype=bool)
    bounds = {"boundary-label": fe.Boundary(u, mask=mask)}

    v = fe.dof.apply(u, bounds, dof0=None)
    assert np.allclose(u.values, v)


def test_loadcase():

    for u in [pre1d(), pre2d(), pre3d()]:
        v = fe.FieldMixed((u, deepcopy(u)))

        ux = fe.dof.uniaxial(u, right=1.0, move=0.2, clamped=False)
        assert len(ux) == 4

        ux = fe.dof.uniaxial(u, right=1.0, move=0.2, clamped=True)
        assert len(ux) == 4
        assert "right" in ux[0]

        ux = fe.dof.uniaxial(u, right=2.0, move=0.2, clamped=True)
        assert len(ux) == 4
        assert "right" in ux[0]

        ux = fe.dof.uniaxial(v, right=1.0, move=0.2, clamped=True)
        assert len(ux) == 5
        assert "right" in ux[0]

        bx = fe.dof.biaxial(u, right=1.0, move=0.2, clamped=False)
        assert len(bx) == 4

        bx = fe.dof.biaxial(u, right=1.0, move=0.2, clamped=True)
        assert len(bx) == 4
        assert "right" in bx[0]

        bx = fe.dof.biaxial(u, right=2.0, move=0.2, clamped=True)
        assert len(bx) == 4
        assert "right" in bx[0]

        bx = fe.dof.biaxial(v, right=1.0, move=0.2, clamped=True)
        assert len(bx) == 5
        assert "right" in bx[0]

        ps = fe.dof.planar(u, right=1.0, move=0.2, clamped=False)
        assert len(ps) == 4

        ps = fe.dof.planar(u, right=1.0, move=0.2, clamped=True)
        assert len(ps) == 4
        assert "right" in ps[0]

        ps = fe.dof.planar(u, right=2.0, move=0.2, clamped=True)
        assert len(ps) == 4
        assert "right" in ps[0]

        ps = fe.dof.planar(v, right=1.0, move=0.2, clamped=True)
        assert len(ps) == 5
        assert "right" in ps[0]

        sh = fe.dof.shear(u, bottom=0.0, top=1.0, move=0.2, sym=True)
        assert len(sh) == 4
        assert "top" in sh[0]

        sh = fe.dof.shear(v, bottom=0.0, top=1.0, move=0.2, sym=False)
        assert len(sh) == 5
        assert "top" in sh[0]


def test_mpc():

    mesh = fe.Cube(n=3)
    element = fe.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

    mesh.points = np.vstack((mesh.points, [2, 0, 0]))
    mesh.update(mesh.cells)

    region = fe.Region(mesh, element, quadrature)

    u = fe.Field(region, dim=3)
    F = u.extract()

    umat = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fe.Boundary(u, fx=f0)
    boundaries["right"] = fe.Boundary(u, fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(u, fx=f2, skip=(0, 1, 1), value=0.5)

    mpc = fe.Boundary(u, fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fe.MultiPointConstraint(mesh, points=mpc, centerpoint=cpoint)
    K_RBE2 = RBE2.stiffness()
    r_RBE2 = RBE2.residuals(u)

    linearform = fe.IntegralForm(umat.gradient(F), u, region.dV, grad_v=True)
    r = linearform.assemble() + r_RBE2

    bilinearform = fe.IntegralForm(
        umat.hessian(F), u, region.dV, u, grad_v=True, grad_u=True
    )
    K = bilinearform.assemble() + K_RBE2

    assert r.shape == (84, 1)
    assert K.shape == (84, 84)


def pre_mpc_mixed(point, values):

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

    mesh.points = np.vstack((mesh.points, point))
    mesh.update(mesh.cells)

    region = fe.Region(mesh, element, quadrature)
    dV = region.dV

    mesh0 = fe.mesh.convert(mesh, order=0)
    element0 = fe.element.ConstantHexahedron()
    region0 = fe.Region(mesh0, element0, quadrature, grad=False)

    displacement = fe.Field(region, dim=3)
    pressure = fe.Field(region0)
    volumeratio = fe.Field(region0, values=1)

    displacement.values[-1] = values

    fields = fe.FieldMixed((displacement, pressure, volumeratio))

    F, p, J = fields.extract()

    nh = fe.NeoHooke(mu=1.0, bulk=2.0)
    umat = fe.ThreeFieldVariation(nh)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fe.Boundary(displacement, fx=f0)
    boundaries["right"] = fe.Boundary(displacement, fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(displacement, fx=f2, skip=(0, 1, 1), value=0.5)

    mpc = fe.Boundary(displacement, fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fe.MultiPointConstraint(mesh, points=mpc, centerpoint=cpoint)
    CONT = fe.MultiPointContact(mesh, points=mpc, centerpoint=cpoint)
    K_RBE2 = RBE2.stiffness()
    r_RBE2 = RBE2.residuals(displacement)

    K_CONT = CONT.stiffness(displacement)
    r_CONT = CONT.residuals(displacement)

    assert K_RBE2.shape == K_CONT.shape
    assert r_RBE2.shape == r_CONT.shape

    linearform = fe.IntegralFormMixed(umat.gradient(F, p, J), fields, dV)
    r = linearform.assemble()

    r_RBE2.resize(*r.shape)
    r = r + r_RBE2

    bilinearform = fe.IntegralFormMixed(umat.hessian(F, p, J), fields, dV, fields)
    K = bilinearform.assemble()

    K_RBE2.resize(*K.shape)
    K = K + K_RBE2


def test_mpc_mixed():
    pre_mpc_mixed(point=[2, 0, 0], values=[0, 0, 0])
    pre_mpc_mixed(point=[2, 0, 0], values=[-5, 0, 0])


if __name__ == "__main__":
    test_boundary()
    test_loadcase()
    test_mpc()
    test_mpc_mixed()

# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:41:27 2021

@author: adutz
"""

import numpy as np
import felupe as fe
import casadi as ca


def test_hex8_nh_rbe2():

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

    mesh.points = np.vstack((mesh.points, [2, 0, 0]))
    mesh.update(mesh.cells)

    region = fe.Region(mesh, element, quadrature)
    dV = region.dV

    displacement = fe.Field(region, dim=3)
    F = fe.math.defgrad(displacement)

    umat = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fe.Boundary(displacement, fx=f0)
    boundaries["right"] = fe.Boundary(displacement, fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(displacement, fx=f2, skip=(0, 1, 1), value=0.5)

    mpc = fe.Boundary(displacement, fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fe.doftools.MultiPointConstraint(mesh, points=mpc, centerpoint=cpoint)
    K_RBE2 = RBE2.stiffness()
    r_RBE2 = RBE2.residuals(displacement)

    linearform = fe.IntegralForm(umat.P(F), displacement, dV, grad_v=True)
    r = linearform.assemble() + r_RBE2

    bilinearform = fe.IntegralForm(
        umat.A(F), displacement, dV, displacement, grad_v=True, grad_u=True
    )
    K = bilinearform.assemble() + K_RBE2

    assert r.shape == (84, 1)
    assert K.shape == (84, 84)


def test_hex8_nh_rbe2_mixed():

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

    mesh.points = np.vstack((mesh.points, [2, 0, 0]))
    mesh.update(mesh.cells)

    region = fe.Region(mesh, element, quadrature)
    dV = region.dV

    mesh0 = fe.mesh.convert(mesh, order=0)
    element0 = fe.element.ConstantHexahedron()
    region0 = fe.Region(mesh0, element0, quadrature)

    displacement = fe.Field(region, dim=3)
    pressure = fe.Field(region0)
    volumeratio = fe.Field(region0, values=1)

    fields = fe.FieldMixed((displacement, pressure, volumeratio))

    F, p, J = fields.extract()

    nh = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)
    umat = fe.constitution.GeneralizedThreeField(nh.P, nh.A)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)
    f2 = lambda x: np.isclose(x, 2)

    boundaries = {}
    boundaries["left"] = fe.Boundary(displacement, fx=f0)
    boundaries["right"] = fe.Boundary(displacement, fx=f2, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(displacement, fx=f2, skip=(0, 1, 1), value=0.5)

    mpc = fe.Boundary(displacement, fx=f1).points
    cpoint = mesh.npoints - 1

    RBE2 = fe.doftools.MultiPointConstraint(mesh, points=mpc, centerpoint=cpoint)
    CONT = fe.doftools.MultiPointContact(mesh, points=mpc, centerpoint=cpoint)
    K_RBE2 = RBE2.stiffness()
    r_RBE2 = RBE2.residuals(displacement)

    K_CONT = CONT.stiffness(displacement)
    r_CONT = CONT.residuals(displacement)

    assert K_RBE2.shape == K_CONT.shape
    assert r_RBE2.shape == r_CONT.shape

    linearform = fe.IntegralFormMixed(umat.f(F, p, J), fields, dV)
    r = linearform.assemble()

    r_RBE2.resize(*r.shape)
    r = r + r_RBE2

    bilinearform = fe.IntegralFormMixed(umat.A(F, p, J), fields, dV, fields)
    K = bilinearform.assemble()

    K_RBE2.resize(*K.shape)
    K = K + K_RBE2


if __name__ == "__main__":
    test_hex8_nh_rbe2()
    test_hex8_nh_rbe2_mixed()

# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:41:27 2021

@author: adutz
"""

import numpy as np
import felupe as fe
import casadi as ca


def test_hex8_nh_nr():

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

    region = fe.Region(mesh, element, quadrature)
    dV = region.dV

    displacement = fe.Field(region, dim=3)

    umat = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)

    boundaries = {}
    boundaries["left"] = fe.Boundary(displacement, fx=f0)
    boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=0.5)

    dof0, dof1 = fe.doftools.partition(displacement, boundaries)
    u0ext = fe.doftools.apply(displacement, boundaries, dof0)

    def fun(F):
        linearform = fe.IntegralForm(umat.P(F), displacement, dV, grad_v=True)
        return linearform.assemble().toarray()[:, 0]

    def jac(F):
        bilinearform = fe.IntegralForm(
            umat.A(F), displacement, dV, displacement, grad_v=True, grad_u=True
        )
        return bilinearform.assemble()

    res = fe.tools.newtonrhapson(
        fun,
        displacement,
        jac,
        solve=fe.tools.solve,
        pre=fe.tools.defgrad,
        update=fe.tools.update,
        check=fe.tools.check,
        kwargs_solve={"field": displacement, "ext": u0ext, "dof0": dof0, "dof1": dof1},
        kwargs_check={
            "tol_f": 1e-3,
            "tol_x": 1e-3,
            "dof0": dof0,
            "dof1": dof1,
            "verbose": 0,
        },
    )

    displacement = res.x

    fe.utils.save(region, displacement, filename=None)


def test_hex8_nh_nr_ad():

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

    region = fe.Region(mesh, element, quadrature)
    dV = region.dV

    displacement = fe.Field(region, dim=3)

    def W(F, mu, bulk):
        "Strain energy density function for Neo-Hookean material formulation."

        J = ca.det(F)
        C = F.T @ F
        I_C = ca.trace(C)
        return mu / 2 * (I_C * J ** (-2 / 3) - 3) + bulk / 2 * (J - 1) ** 2

    umat = fe.constitution.ad.Material(W, mu=1.0, bulk=2.0)

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)

    boundaries = {}
    boundaries["left"] = fe.Boundary(displacement, fx=f0)
    boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=0.5)

    dof0, dof1 = fe.doftools.partition(displacement, boundaries)
    u0ext = fe.doftools.apply(displacement, boundaries, dof0)

    def fun(F):
        linearform = fe.IntegralForm(umat.f(F), displacement, dV, grad_v=True)
        return linearform.assemble().toarray()[:, 0]

    def jac(F):
        bilinearform = fe.IntegralForm(
            umat.A(F), displacement, dV, displacement, grad_v=True, grad_u=True
        )
        return bilinearform.assemble()

    res = fe.tools.newtonrhapson(
        fun,
        displacement,
        jac,
        solve=fe.tools.solve,
        pre=fe.tools.defgrad,
        update=fe.tools.update,
        check=fe.tools.check,
        kwargs_solve={"field": displacement, "ext": u0ext, "dof0": dof0, "dof1": dof1},
        kwargs_check={
            "tol_f": 1e-3,
            "tol_x": 1e-3,
            "dof0": dof0,
            "dof1": dof1,
            "verbose": 0,
        },
    )

    displacement = res.x

    fe.utils.save(region, displacement, filename=None)


if __name__ == "__main__":
    test_hex8_nh_nr()
    test_hex8_nh_nr_ad()

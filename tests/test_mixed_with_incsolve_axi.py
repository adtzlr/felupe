# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:49:57 2021

@author: Andreas
"""

import numpy as np
import felupe as fe
import casadi as ca


def test_mixed_with_incsolve_axi():

    mesh = fe.mesh.Rectangle(a=(0, 0.5), b=(1, 1.5), n=11)
    mesh0 = fe.mesh.convert(mesh, order=0)

    element0 = fe.element.ConstantQuad()
    element1 = fe.element.Quad()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=2)

    region0 = fe.Region(mesh0, element0, quadrature)
    region = fe.Region(mesh, element1, quadrature)

    displacement = fe.FieldAxisymmetric(region, dim=2)
    pressure = fe.Field(region0, dim=1)
    volumeratio = fe.Field(region0, dim=1, values=1)
    fields = (displacement, pressure, volumeratio)

    f1 = lambda x: np.isclose(x, 1)

    boundaries = fe.doftools.symmetry(displacement)
    boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(displacement, fx=f1, skip=(0, 1, 1))

    dof0, dof1, unstack = fe.doftools.partition(fields, boundaries)

    neohooke = fe.constitution.models.NeoHooke(mu=1.0, bulk=500.0)
    umat = fe.constitution.variation.upJ(neohooke.P, neohooke.A)

    fe.tools.incsolve(
        fields, region, umat.f, umat.A, boundaries, [-0.1, -0.2], verbose=0
    )

    fe.utils.save(region, fields, unstack=unstack, filename="result.vtk")


if __name__ == "__main__":
    test_mixed_with_incsolve_axi()

# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:41:27 2021

@author: adtzlr
"""

import numpy as np
import felupe as fe

mesh = fe.mesh.Cube(n=9)
element = fe.element.Hexahedron()
quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

region = fe.Region(mesh, element, quadrature)

dV = region.dV
V = dV.sum()

displacement = fe.Field(region, dim=3)

u = displacement.values
ui = displacement.interpolate()
dudX = displacement.grad()

from felupe.math import identity

F = identity(dudX) + dudX

from casadi import det, transpose, trace


def W(F, mu, bulk):
    "Neo-Hooke"

    J = det(F)
    C = transpose(F) @ F

    return mu / 2 * (J ** (-2 / 3) * trace(C) - 3) + bulk / 2 * (J - 1) ** 2


umat = fe.constitution.StrainEnergyDensity(W, mu=1.0, bulk=2.0)

P = umat.P
A = umat.A

f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

boundaries = {}
boundaries["left"] = fe.Boundary(displacement, fx=f0)
boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
boundaries["move"] = fe.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=0.5)

dof0, dof1 = fe.doftools.partition(displacement, boundaries)
u0ext = fe.doftools.apply(displacement, boundaries, dof0)

linearform = fe.IntegralForm(P(F), displacement, dV, grad_v=True)
bilinearform = fe.IntegralForm(
    A(F), displacement, dV, u=displacement, grad_v=True, grad_u=True
)

r = linearform.assemble().toarray()[:, 0]
K = bilinearform.assemble()

system = fe.solve.partition(displacement, K, dof1, dof0, r)
du = fe.solve.solve(*system, u0ext).reshape(*u.shape)
# displacement += du

for iteration in range(8):
    dudX = displacement.grad()
    F = identity(dudX) + dudX

    linearform = fe.IntegralForm(P(F), displacement, dV, grad_v=True)
    bilinearform = fe.IntegralForm(
        A(F), displacement, dV, u=displacement, grad_v=True, grad_u=True
    )

    r = linearform.assemble().toarray()[:, 0]
    K = bilinearform.assemble()

    system = fe.solve.partition(displacement, K, dof1, dof0, r)
    du = fe.solve.solve(*system, u0ext).reshape(*u.shape)

    norm = np.linalg.norm(du)
    print(iteration, norm)
    displacement += du

    if norm < 1e-12:
        break

F[:, :, 0, 0]

fe.utils.save(region, displacement, filename="result.vtk")

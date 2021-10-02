# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:41:27 2021

@author: adtzlr
"""

import numpy as np
import felupe as fe
import meshzoo

try:
    from pypardiso import spsolve

except ImportError:
    print("skip PyPardiso")
    from scipy.sparse.linalg import spsolve

points_p, cells_p = meshzoo.cube_tetra(a0=(0, 0, 0), a1=(1, 1, 1), n=6)
points_p, cells_p = fe.mesh.fix(points_p, cells_p, "tetra")
points_u, cells_u, _ = fe.mesh.add_midpoints_volumes(points_p, cells_p, "tetra")

mesh_u = fe.mesh.Mesh(points_u, cells_u, "tetra4+1")
mesh_p = fe.mesh.Mesh(points_p, cells_p, "tetra")

element_u = fe.element.TetraMINI()
element_p = fe.element.Tetra()

quadrature = fe.quadrature.Tetrahedron(order=2)

region_u = fe.Region(mesh_u, element_u, quadrature)
region_p = fe.Region(mesh_p, element_p, quadrature)

displacement = fe.Field(region_u, dim=3)
pressure = fe.Field(region_p)
field = fe.FieldMixed((displacement, pressure))

boundaries, dof0, dof1, unstack, u0ext = fe.doftools.uniaxial(field)

from casadi import det, transpose, trace


def W(F, p, mu, bulk):
    """ "Strain energy density function of Neo-Hooke material formulation
    as perturbed lagrangian."""

    J = det(F)
    Fu = J ** (-1 / 3) * F
    Cu = transpose(Fu) @ Fu

    return mu / 2 * (trace(Cu) - 3) + p * (J - 1) - 1 / (2 * bulk) * p ** 2


umat = fe.constitution.StrainEnergyDensityTwoField(W, mu=1.0, bulk=5000.0)

for increment, move in enumerate([-0.1, -0.2, -0.3]):

    boundaries["move"].value = move
    u0ext = fe.doftools.apply(displacement, boundaries, dof0)

    print("\nIncrement %d" % (1 + increment))
    print("===========")

    print("\nPrescribed displacements on `move` = %1.2f \n" % move)
    print("#   norm(u)    norm(p)")
    print("------------------------")

    for iteration in range(15):

        F, p = field.extract()

        f = umat.f(F, p, False)
        A = umat.A(F, p, False)

        linearform = fe.IntegralFormMixed(fun=f, v=field, dV=region_u.dV)
        bilinearform = fe.IntegralFormMixed(fun=A, v=field, dV=region_u.dV, u=field)

        r = linearform.assemble().toarray()[:, 0]
        K = bilinearform.assemble()

        system = fe.solve.partition(field, K, dof1, dof0, r)
        dfield = np.split(fe.solve.solve(*system, u0ext, solver=spsolve), unstack)

        field = fe.tools.update(field, dfield)
        norms = fe.math.norms(dfield)

        print(1 + iteration, *["%1.4e" % norm for norm in norms])

        if np.all(norms < 1e-10):
            print(
                "\nReaction Force X on `move` = %1.4g" % r[boundaries["move"].dof].sum()
            )
            break

point_data = {
    "p": pressure.values,
}

displacement.values = displacement.values[: len(points_p)]

unstack2 = np.array([points_p.size, points_p.size + cells_p.size])

fe.utils.save(
    region_p, field, unstack=unstack2, point_data=point_data, filename="result.vtk"
)

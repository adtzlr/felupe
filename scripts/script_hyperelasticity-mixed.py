# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:41:27 2021

@author: adtzlr
"""

import numpy as np
import felupe as fe

try:
    from pypardiso import spsolve

except ImportError:
    print("skip PyPardiso")
    from scipy.sparse.linalg import spsolve

mesh = fe.mesh.Cube(n=11)
mesh_p = fe.mesh.convert(mesh, order=0)

element = fe.element.Hexahedron()
element_p = fe.element.ConstantHexahedron()

quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)

region = fe.Region(mesh, element, quadrature)
region_p = fe.Region(mesh_p, element_p, quadrature)

displacement = fe.Field(region, dim=3)
pressure = fe.Field(region_p)
field = fe.FieldMixed((displacement, pressure))

boundaries, dof0, dof1, unstack, u0ext = fe.doftools.uniaxial(field)

from casadi import det, transpose, trace


def W(F, p, mu, bulk):
    """ "Strain energy density function of Neo-Hooke material formulation
    as perturbed lagrangian."""

    J = det(F)
    C = transpose(F) @ F

    return (
        mu / 2 * (J ** (-2 / 3) * trace(C) - 3) + p * (J - 1) - 1 / (2 * bulk) * p ** 2
    )


umat = fe.constitution.StrainEnergyDensityTwoField(W, mu=1.0, bulk=5000.0)

for increment, move in enumerate([-0.1, -0.2, -0.3, -0.4]):

    boundaries["move"].value = move
    u0ext = fe.doftools.apply(displacement, boundaries, dof0)

    print("\nIncrement %d" % (1 + increment))
    print("===========")

    print("\nPrescribed displacements on `move` = %1.2f \n" % move)
    print("#   norm(u)    norm(p)")
    print("------------------------")

    for iteration in range(15):

        F, p = field.extract()

        f = umat.f(F, p)
        A = umat.A(F, p)

        linearform = fe.IntegralFormMixed(fun=f, v=field, dV=region.dV)
        bilinearform = fe.IntegralFormMixed(fun=A, v=field, dV=region.dV, u=field)

        r = linearform.assemble(parallel=True).toarray()[:, 0]
        K = bilinearform.assemble(parallel=True)

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

cell_data = {
    "pressure": [
        pressure.values,
    ],
}

fe.utils.save(
    region,
    field,
    r,
    K,
    F,
    f,
    A,
    unstack=unstack,
    cell_data=cell_data,
    filename="result.vtk",
)

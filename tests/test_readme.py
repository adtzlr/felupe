# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:33:53 2021

@author: adutz
"""

import numpy as np


def test_readme():

    import felupe

    mesh = felupe.Cube(n=9)
    element = felupe.Hexahedron()
    quadrature = felupe.GaussLegendre(order=1, dim=3)

    region = felupe.Region(mesh, element, quadrature)

    dV = region.dV
    V = dV.sum()

    displacement = felupe.Field(region, dim=3)

    u = displacement.values
    ui = displacement.interpolate()
    dudX = displacement.grad()

    field = felupe.FieldContainer([displacement])

    F = field.extract(grad=True, sym=False, add_identity=True)

    umat = felupe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    P = umat.gradient
    A = umat.hessian

    import numpy as np

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)

    boundaries = {}
    boundaries["left"] = felupe.Boundary(displacement, fx=f0)
    boundaries["right"] = felupe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
    boundaries["move"] = felupe.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=0.5)

    dof0, dof1 = felupe.dof.partition(field, boundaries)
    ext0 = felupe.dof.apply(field, boundaries, dof0)

    linearform = felupe.IntegralForm(P(F)[:-1], field, dV)
    bilinearform = felupe.IntegralForm(A(F), field, dV, field)

    r = linearform.assemble().toarray()[:, 0]
    K = bilinearform.assemble()

    from scipy.sparse.linalg import spsolve  # default

    # from pypardiso import spsolve

    system = felupe.solve.partition(field, K, dof1, dof0, r)
    dfield = felupe.solve.solve(*system, ext0, solver=spsolve).reshape(*u.shape)

    # du = np.split(dfield, offsets)
    # field += du

    for iteration in range(8):
        F = field.extract()

        linearform = felupe.IntegralForm(P(F)[:-1], field, dV)
        bilinearform = felupe.IntegralForm(A(F), field, dV, field)

        r = linearform.assemble()
        K = bilinearform.assemble()

        system = felupe.solve.partition(field, K, dof1, dof0, r)
        dfield = felupe.solve.solve(*system, ext0, solver=spsolve).reshape(*u.shape)

        du = np.split(dfield, field.offsets)

        norm = felupe.math.norm(du)
        print(iteration, norm[0])
        field += du

        if iteration == 0:
            assert np.round(norm[0], 5) == 8.17418

        if norm[0] < 1e-12:
            break

    F[0][:, :, 0, 0]

    felupe.save(region, field, filename="result.vtk")

    from felupe.math import dot, det, transpose

    PK1 = P(F)[0]
    F = F[0]

    s = dot(PK1, transpose(F)) / det(F)

    # stress shifted and averaged to mesh-points
    cauchy_shifted = felupe.topoints(s, region, sym=True, mode="tensor")

    # stress projected and averaged to mesh-points
    cauchy_projected = felupe.project(s, region)

    felupe.save(
        region,
        field,
        filename="result_with_cauchy.vtk",
        point_data={
            "CauchyStressProjected": cauchy_projected,
            "CauchyStressShifted": cauchy_shifted,
        },
    )


if __name__ == "__main__":
    test_readme()

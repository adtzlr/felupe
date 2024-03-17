# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:33:53 2021

@author: adutz
"""

import numpy as np

import felupe as fem


def test_readme():
    mesh = fem.Cube(n=9)
    element = fem.Hexahedron()
    quadrature = fem.GaussLegendre(order=1, dim=3)

    region = fem.Region(mesh, element, quadrature)

    dV = region.dV
    V = dV.sum()

    displacement = fem.Field(region, dim=3)

    u = displacement.values
    ui = displacement.interpolate()
    dudX = displacement.grad()

    field = fem.FieldContainer([displacement])

    F = field.extract(grad=True, sym=False, add_identity=True)

    umat = fem.constitution.NeoHooke(mu=1.0, bulk=2.0)

    P = umat.gradient
    A = umat.hessian

    import numpy as np

    f0 = lambda x: np.isclose(x, 0)
    f1 = lambda x: np.isclose(x, 1)

    boundaries = {}
    boundaries["left"] = fem.Boundary(displacement, fx=f0)
    boundaries["right"] = fem.Boundary(displacement, fx=f1, skip=(1, 0, 0))
    boundaries["move"] = fem.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=0.5)

    dof0, dof1 = fem.dof.partition(field, boundaries)
    ext0 = fem.dof.apply(field, boundaries, dof0)

    linearform = fem.IntegralForm(P(F)[:-1], field, dV)
    bilinearform = fem.IntegralForm(A(F), field, dV, field)

    r = linearform.assemble().toarray()[:, 0]
    K = bilinearform.assemble()

    from scipy.sparse.linalg import spsolve  # default

    # from pypardiso import spsolve

    system = fem.solve.partition(field, K, dof1, dof0, r)
    dfield = fem.solve.solve(*system, ext0, solver=spsolve).reshape(*u.shape)

    # du = np.split(dfield, offsets)
    # field += du

    for iteration in range(8):
        F = field.extract()

        linearform = fem.IntegralForm(P(F)[:-1], field, dV)
        bilinearform = fem.IntegralForm(A(F), field, dV, field)

        r = linearform.assemble()
        K = bilinearform.assemble()

        system = fem.solve.partition(field, K, dof1, dof0, r)
        dfield = fem.solve.solve(*system, ext0, solver=spsolve).reshape(*u.shape)

        du = np.split(dfield, field.offsets)

        norm = fem.math.norm(du)
        print(iteration, norm[0])
        field += du

        if iteration == 0:
            assert np.round(norm[0], 5) == 8.17418

        if norm[0] < 1e-12:
            break

    F[0][:, :, 0, 0]

    fem.save(region, field, filename="result.vtk")

    from felupe.math import det, dot, transpose

    PK1 = P(F)[0]
    F = F[0]

    s = dot(PK1, transpose(F)) / det(F)

    # stress shifted and averaged to mesh-points
    cauchy_shifted = fem.topoints(fem.math.tovoigt(s), region)

    # stress projected and averaged to mesh-points
    cauchy_projected = fem.project(s, region)

    fem.save(
        region,
        field,
        filename="result_with_cauchy.vtk",
        point_data={
            "CauchyStressProjected": cauchy_projected,
            "CauchyStressShifted": cauchy_shifted,
        },
    )


def test_readme_form():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

    from felupe.math import ddot, grad, sym, trace

    @fem.Form(v=field, u=field)
    def bilinearform():
        def a(v, u, μ=1.0, λ=2.0):
            δε, ε = sym(grad(v)), sym(grad(u))
            return 2 * μ * ddot(δε, ε) + λ * trace(δε) * trace(ε)

        return [a]

    @fem.Form(v=field)
    def linearform():
        def L(v, μ=1.0, λ=2.0):
            δε = sym(grad(v))
            ε = field.extract(grad=True, sym=True, add_identity=False)[0]
            return 2 * μ * ddot(δε, ε) + λ * trace(δε) * trace(ε)

        return [L]

    item = fem.FormItem(bilinearform)
    step = fem.Step(items=[item], boundaries=boundaries)
    fem.Job(steps=[step]).evaluate()

    field[0].fill(0)

    item = fem.FormItem(bilinearform, linearform)
    step = fem.Step(items=[item], boundaries=boundaries)
    fem.Job(steps=[step]).evaluate()

    item = fem.FormItem(linearform=linearform)
    item.assemble.vector(field)
    assert item.assemble.matrix(field).nnz == 0

    item = fem.FormItem()
    item.assemble.vector(field)
    item.assemble.matrix(field)
    assert item.assemble.vector(field).nnz == 0
    assert item.assemble.matrix(field).nnz == 0


if __name__ == "__main__":
    test_readme()
    test_readme_form()

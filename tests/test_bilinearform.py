# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import felupe as fem
from felupe.assembly.expression._bilinear import BilinearForm
from felupe.assembly.expression._linear import LinearForm
from felupe.math import dddot, ddot, dot, dya, grad, hess, sym, trace


def lform(v, w):
    return dot(w, v)


def lform_grad(v, F):
    return ddot(F, v)


def lformu(v, F, p):
    return ddot(F, grad(v))


def lformp(q, F, p):
    return dot(p, q)


def a_uu(v, u, F, p):
    return ddot(grad(u), ddot(dya(F, F), grad(v)))


def a_up(v, r, F, p):
    return dot(p, r) * ddot(F, grad(v))


def a_pp(q, r, F, p):
    return dot(p, q) * dot(r, p)


def pre(dim):
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    region0 = fem.RegionConstantHexahedron(mesh.dual(points_per_cell=1))
    u = fem.Field(region, dim=3)
    p = fem.Field(region0)
    up = fem.FieldContainer([u, p])

    return up


def test_form_decorator():
    field = pre(dim=3)
    F, p = field.extract()

    @fem.Form(v=field, u=field)
    def a():
        return (a_uu, a_up, a_pp)

    a.assemble(field, field, kwargs=dict(F=F, p=p))

    @fem.Form(v=field)
    def L():
        return (lformu, lformp)

    L.assemble(field, kwargs=dict(F=F, p=p), parallel=False)
    L.assemble(field, kwargs=dict(F=F, p=p), parallel=True)
    L.assemble(field, kwargs=dict(F=F, p=p), parallel=False, sym=True)
    L.assemble(field, kwargs=dict(F=F, p=p), parallel=True, sym=True)

    @fem.Form(v=field)
    def L():
        return (lformu, lformp)

    L.integrate(field, kwargs=dict(F=F, p=p), parallel=False)
    L.integrate(field, kwargs=dict(F=F, p=p), parallel=True)
    L.integrate(field, kwargs=dict(F=F, p=p), parallel=False, sym=True)
    L.integrate(field, kwargs=dict(F=F, p=p), parallel=True, sym=True)

    L.assemble(field, kwargs=dict(F=F, p=p), parallel=False)
    L.assemble(field, kwargs=dict(F=F, p=p), parallel=True)
    L.assemble(field, kwargs=dict(F=F, p=p), parallel=False, sym=True)
    L.assemble(field, kwargs=dict(F=F, p=p), parallel=True, sym=True)


def test_expression_integrate_uses_nonmutable_kwargs_default():
    assert LinearForm.integrate.__defaults__[0] is None
    assert BilinearForm.integrate.__defaults__[0] is None


def test_linear_elastic():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])

    @fem.Form(v=field, u=field, kwargs={"mu": 1.0, "lmbda": 2.0})
    def bilinearform():
        "A container for a bilinear form."

        def linear_elasticity(v, u, mu, lmbda):
            de, e = sym(grad(v)), sym(grad(u))
            return 2 * mu * ddot(de, e) + lmbda * trace(de) * trace(e)

        return [linear_elasticity]

    bilinearform.integrate(v=field, u=field, parallel=False)
    bilinearform.integrate(v=field, u=field, parallel=True)
    bilinearform.integrate(v=field, u=field, parallel=False, sym=True)
    bilinearform.integrate(v=field, u=field, parallel=True, sym=True)

    bilinearform.assemble(v=field, u=field, parallel=False)
    bilinearform.assemble(v=field, u=field, parallel=True)
    bilinearform.assemble(v=field, u=field, parallel=False, sym=True)
    bilinearform.assemble(v=field, u=field, parallel=True, sym=True)


def test_huhu_regularization():
    mesh = fem.Cube()
    region = fem.RegionHexahedron(mesh, hess=True)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])

    @fem.Form(v=field, u=field)
    def bilinearform():
        return [lambda v, u: dddot(hess(v), hess(u))]

    @fem.Form(v=field)
    def linearform():
        u = field[0]
        return [lambda v: dddot(hess(v), hess(u))]

    matrix = bilinearform.assemble(v=field, u=field)
    vector = linearform.assemble(v=field)

    assert matrix.shape == (24, 24)
    assert vector.shape == (24, 1)
    assert np.allclose(vector.toarray(), 0.0)


def test_huhu_regularization_planestrain():
    mesh = fem.Rectangle()
    region = fem.RegionQuad(mesh, hess=True)
    displacement = fem.FieldPlaneStrain(region, dim=2)
    field = fem.FieldContainer([displacement])

    @fem.Form(v=field, u=field)
    def bilinearform():
        return [lambda v, u: dddot(hess(v), hess(u))]

    @fem.Form(v=field)
    def linearform():
        u = field[0]
        return [lambda v: dddot(hess(v), hess(u)[:2, :2, :2])]

    matrix = bilinearform.assemble(v=field, u=field)
    vector = linearform.assemble(v=field)

    assert matrix.shape == (8, 8)
    assert vector.shape == (8, 1)
    assert np.allclose(vector.toarray(), 0.0)


if __name__ == "__main__":
    test_form_decorator()
    test_linear_elastic()
    test_huhu_regularization()
    test_huhu_regularization_planestrain()

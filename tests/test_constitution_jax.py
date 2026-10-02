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
import jax.numpy as jnp
import numpy as np
import pytest

import felupe as fem
import felupe.constitution.jax as mat


def test_vmap():
    def f(x, a=1.0):
        return x

    def g(x, y, a=1.0, **kwargs):
        return x

    vf = fem.constitution.jax.vmap(f)
    vg = fem.constitution.jax.vmap(g)

    x = (np.eye(3).reshape(1, 1, 3, 3) * np.ones((10, 2, 1, 1))).T

    z = vf(x, a=1.0)

    assert np.allclose(z, vf(x, 1.0))
    assert np.allclose(z, vf(a=1.0, x=x))

    with pytest.raises(TypeError):
        vf(x, a=1.0, b=2.0)

    # does not raise an error because of `g(..., **kwargs)`
    assert np.allclose(z, vg(x, a=1.0, b=2.0))


def test_hyperelastic_jax():
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    md = mat.models.hyperelastic
    for W in [
        md.mooney_rivlin,
        md.yeoh,
        md.third_order_deformation,
        md.miehe_goektepe_lulei,
        md.ogden,
        md.storakers,
        md.van_der_waals,
        md.extended_tube,
        md.blatz_ko,
    ]:
        umat = mat.Hyperelastic(W, **W.kwargs)
        solid = fem.SolidBody(umat=umat, field=field)
        solid.evaluate.gradient()
        solid.evaluate.hessian()

    umat = mat.Hyperelastic(W, **W.kwargs, parallel=True)
    umat = mat.Hyperelastic(W, **W.kwargs, jit=True)


def test_hyperelastic_jax_statevars():
    def W(C, statevars, C10, K):
        I3 = jnp.linalg.det(C)
        J = jnp.sqrt(I3)
        I1 = I3 ** (-1 / 3) * jnp.trace(C)
        statevars_new = statevars.at[0].set(I1)
        return C10 * (I1 - 3) + K * (J - 1) ** 2 / 2, statevars_new

    W.kwargs = {"C10": 0.5}

    umat = mat.Hyperelastic(W, C10=0.5, K=2.0, nstatevars=1, jit=True)
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    solid = fem.SolidBody(umat=umat, field=field)
    solid.evaluate.gradient()
    solid.evaluate.hessian()


def test_material_jax():
    def dWdF(F, C10, K):
        J = jnp.linalg.det(F)
        C = F.T @ F
        Cu = J ** (-2 / 3) * C
        dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)

        P = 2 * C10 * F @ dev(Cu) @ jnp.linalg.inv(C)
        return P + K * (J - 1) * J * jnp.linalg.inv(C)

    umat = mat.Material(dWdF, C10=0.5, K=2.0, parallel=True)
    umat = mat.Material(dWdF, C10=0.5, K=2.0, jit=True)

    for fun in [dWdF, mat.updated_lagrange(dWdF), mat.total_lagrange(dWdF)]:
        umat = mat.Material(fun, C10=0.5, K=2.0)
        mesh = fem.Cube(n=2)
        region = fem.RegionHexahedron(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=3)])

        solid = fem.SolidBody(umat=umat, field=field)
        solid.evaluate.gradient()
        solid.evaluate.hessian()


def test_material_jax_statevars():
    def dWdF(F, statevars, C10, K):
        J = jnp.linalg.det(F)
        C = F.T @ F
        Cu = J ** (-2 / 3) * C
        dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)

        P = 2 * C10 * F @ dev(Cu) @ jnp.linalg.inv(C)
        statevars_new = statevars.at[0].set(J)
        return P + K * (J - 1) * J * jnp.linalg.inv(C), statevars_new

    dWdF.kwargs = {"C10": 0.5}

    for fun in [dWdF, mat.updated_lagrange(dWdF), mat.total_lagrange(dWdF)]:
        umat = mat.Material(fun, C10=0.5, K=2.0, nstatevars=1, jit=True)
        mesh = fem.Cube(n=2)
        region = fem.RegionHexahedron(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=3)])

        solid = fem.SolidBody(umat=umat, field=field)
        solid.evaluate.gradient()
        solid.evaluate.hessian()


def test_material_included_jax_statevars():
    for fun, nstatevars in zip(
        [
            mat.models.lagrange.becker,
            mat.models.lagrange.morph,
            mat.models.lagrange.morph_representative_directions,
        ],
        [0, 13, 84],
    ):
        umat = mat.Material(
            fun,
            **fun.kwargs,
            nstatevars=nstatevars,
        )
        mesh = fem.Cube(n=2)
        region = fem.RegionHexahedron(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=3)])

        solid = fem.SolidBody(umat=umat, field=field)
        solid.evaluate.gradient()
        solid.evaluate.hessian()


def test_eigvalsh_perturbation():
    import jax

    from felupe.constitution.jax._helpers import PERTURBATION, eigvalsh, perturb

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)

    try:
        # default relative perturbation: sqrt(machine eps) in double precision (as in
        # tensortrax) and 1e-4 in single precision
        cases = [(jnp.float32, 1e-4), (jnp.float64, np.sqrt(np.finfo(float).eps))]
        for dtype, eps in cases:
            A = 2 * jnp.eye(3, dtype=dtype)
            dA = perturb(A) - A
            assert dA.dtype == dtype
            ref = eps * np.linalg.norm(A) * PERTURBATION
            assert np.allclose(dA, ref, rtol=0, atol=1e-2 * eps * np.linalg.norm(A))

        eps = np.sqrt(np.finfo(float).eps)
        ogden = lambda C: jnp.sum(jnp.linalg.det(C) ** (-1 / 3) * eigvalsh(C))
        neo_hooke = lambda C: jnp.linalg.det(C) ** (-1 / 3) * jnp.trace(C)  # = ogden

        # repeated eigenvalues, distinct eigenvector along a (uniaxial loading),
        # a = (1, +/-1, 0) is uniaxial loading at 45° in the xy-plane
        for a in [[1, -1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]]:
            a = jnp.array(a, dtype=float) / jnp.linalg.norm(jnp.array(a, dtype=float))
            C = (2.25 - 1 / 1.5) * jnp.outer(a, a) + jnp.eye(3) / 1.5

            # the repeated eigenvalues are separated by the perturbation
            λ = eigvalsh(C)
            assert np.min(np.diff(λ)) > 0.5 * eps * np.linalg.norm(C)

            H = jax.hessian(ogden)(C)
            assert np.allclose(H, jax.hessian(neo_hooke)(C), atol=1e-7)

        # the perturbation is relative to the magnitude of the tensor
        for scale in [1e-6, 1e6]:
            λ = eigvalsh(scale * C)
            assert np.allclose(λ, np.linalg.eigvalsh(scale * C), rtol=1e-7, atol=0)

    finally:
        jax.config.update("jax_enable_x64", x64)


def test_morph_jax_tensortrax():
    import jax

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)

    try:
        p = [0.039, 0.371, 0.174, 2.41, 0.0094, 6.84, 5.65, 0.244]
        umat_jax = mat.Material(mat.models.lagrange.morph, p=p, nstatevars=13)
        umat_ttx = fem.MaterialAD(fem.morph, p=p, nstatevars=13)

        statevars = np.zeros((13, 1, 1))
        statevars[[1, 4, 6]] = 1.0  # Cn = 1 (upper triangle entries)

        # uniaxial loading at 45° in the xy-plane (repeated eigenvalues)
        a = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        U = np.eye(3) / np.sqrt(1.5) + (1.5 - 1 / np.sqrt(1.5)) * np.outer(a, a)

        for F in [np.eye(3), U]:
            x = [F.reshape(3, 3, 1, 1), statevars]

            P_jax, statevars_jax = umat_jax.gradient(x)
            P_ttx, statevars_ttx = umat_ttx.gradient(x)

            assert np.allclose(P_jax, P_ttx, rtol=1e-6, atol=1e-8)
            assert np.allclose(statevars_jax, statevars_ttx, rtol=1e-6, atol=1e-8)

            A_jax = umat_jax.hessian(x)[0]
            A_ttx = umat_ttx.hessian(x)[0]

            assert np.allclose(A_jax, A_ttx, rtol=1e-6, atol=1e-8)

    finally:
        jax.config.update("jax_enable_x64", x64)


def test_eigenvalue_models_jax_tensortrax():
    import jax

    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)

    try:
        models = [
            (
                mat.Hyperelastic(
                    mat.models.hyperelastic.ogden, mu=[1.0, 0.2], alpha=[2.5, -1.5]
                ),
                fem.Hyperelastic(fem.ogden, mu=[1.0, 0.2], alpha=[2.5, -1.5]),
            ),
            (
                mat.Hyperelastic(
                    mat.models.hyperelastic.extended_tube,
                    Gc=0.1867,
                    Ge=0.2169,
                    beta=0.2,
                    delta=0.09693,
                ),
                fem.Hyperelastic(
                    fem.extended_tube, Gc=0.1867, Ge=0.2169, beta=0.2, delta=0.09693
                ),
            ),
            (
                mat.Hyperelastic(
                    mat.models.hyperelastic.storakers,
                    mu=[4.5 * (1.85 / 2), -4.5 * (-9.2 / 2)],
                    alpha=[1.85, -9.2],
                    beta=[0.92, 0.92],
                ),
                fem.Hyperelastic(
                    fem.storakers,
                    mu=[4.5 * (1.85 / 2), -4.5 * (-9.2 / 2)],
                    alpha=[1.85, -9.2],
                    beta=[0.92, 0.92],
                ),
            ),
            (
                mat.Material(mat.models.lagrange.becker, mu=1.0, lmbda=2.0),
                fem.MaterialAD(fem.becker, mu=1.0, lmbda=2.0),
            ),
        ]

        # uniaxial loading at 45° in the xy-plane (repeated eigenvalues)
        a = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        U = np.eye(3) / np.sqrt(1.5) + (1.5 - 1 / np.sqrt(1.5)) * np.outer(a, a)

        for umat_jax, umat_ttx in models:
            for F in [np.eye(3), U]:
                x = [F.reshape(3, 3, 1, 1), None]

                P_jax = umat_jax.gradient(x)[0]
                P_ttx = umat_ttx.gradient(x)[0]

                A_jax = umat_jax.hessian(x)[0]
                A_ttx = umat_ttx.hessian(x)[0]

                atol = 1e-7 * np.abs(A_ttx).max()
                assert np.allclose(P_jax, P_ttx, rtol=1e-6, atol=atol)
                assert np.allclose(A_jax, A_ttx, rtol=1e-6, atol=atol)

    finally:
        jax.config.update("jax_enable_x64", x64)


if __name__ == "__main__":
    test_vmap()
    test_hyperelastic_jax()
    test_hyperelastic_jax_statevars()
    test_material_jax()
    test_material_jax_statevars()
    test_material_included_jax_statevars()
    test_eigvalsh_perturbation()
    test_morph_jax_tensortrax()
    test_eigenvalue_models_jax_tensortrax()

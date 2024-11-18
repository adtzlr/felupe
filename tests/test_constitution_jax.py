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
            mat.models.lagrange.morph,
            mat.models.lagrange.morph_representative_directions,
        ],
        [13, 84],
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


if __name__ == "__main__":
    test_vmap()
    test_hyperelastic_jax()
    test_hyperelastic_jax_statevars()
    test_material_jax()
    test_material_jax_statevars()
    test_material_included_jax_statevars()

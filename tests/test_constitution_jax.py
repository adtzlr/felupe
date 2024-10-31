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
import numpy as np
import pytest

import felupe as fem


def test_vmap():
    def f(x, a=1.0):
        return x

    def g(x, y, a=1.0, **kwargs):
        return x

    vf = fem.constitution.jax.vmap(f)
    vg = fem.constitution.jax.vmap(g)

    x = np.eye(3).reshape(1, 3, 3) * np.ones((10, 1, 1))

    z = vf(x, a=1.0)

    assert np.allclose(z, vf(x, 1.0))
    assert np.allclose(z, vf(a=1.0, x=x))

    with pytest.raises(TypeError):
        vf(x, a=1.0, b=2.0)

    # does not raise an error because of `g(..., **kwargs)`
    assert np.allclose(z, vg(x, a=1.0, b=2.0))


def test_hyperelastic_jax():
    import jax.numpy as jnp

    def W(C, C10, K):
        I3 = jnp.linalg.det(C)
        J = jnp.sqrt(I3)
        I1 = I3 ** (-1 / 3) * jnp.trace(C)
        return C10 * (I1 - 3) + K * (J - 1) ** 2 / 2

    umat = fem.constitution.jax.Hyperelastic(W, C10=0.5, K=2.0, jit=True)
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(umat=umat, field=field)

    move = fem.math.linsteps([0, 1], num=3)
    ramp = {boundaries["move"]: move}
    step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
    job = fem.Job(steps=[step])
    job.evaluate(tol=1e-4)


def test_hyperelastic_jax_statevars():
    import jax.numpy as jnp

    def W(C, statevars, C10, K):
        I3 = jnp.linalg.det(C)
        J = jnp.sqrt(I3)
        I1 = I3 ** (-1 / 3) * jnp.trace(C)
        statevars_new = I1
        return C10 * (I1 - 3) + K * (J - 1) ** 2 / 2, statevars_new

    umat = fem.constitution.jax.Hyperelastic(W, C10=0.5, K=2.0, nstatevars=1, jit=True)
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(umat=umat, field=field)

    move = fem.math.linsteps([0, 1], num=3)
    ramp = {boundaries["move"]: move}
    step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
    job = fem.Job(steps=[step])
    job.evaluate(tol=1e-4)


if __name__ == "__main__":
    test_vmap()
    test_hyperelastic_jax()
    test_hyperelastic_jax_statevars()

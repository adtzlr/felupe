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

import felupe as fem


def easy():
    "A moderately stretched cube (plain Newton converges quadratically)."
    region = fem.RegionHexahedron(fem.Cube(n=4))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )
    umat = fem.NeoHooke(mu=1.0, bulk=2.0)
    solid = fem.SolidBody(umat=umat, field=field)
    return field, solid, loadcase


def test_newton_plugins_unchanged():
    "Plugins without modifications don't change the results of Newton's method."

    results = []
    for plugins in [None, [], [fem.Plugin()]]:
        field, solid, loadcase = easy()
        res = fem.newtonraphson(items=[solid], verbose=0, plugins=plugins, **loadcase)
        results.append(res)

    ref = results[0]
    assert ref.success
    assert ref.iterations == 4

    for res in results[1:]:
        assert res.iterations == ref.iterations
        assert np.array_equal(res.fun, ref.fun)
        assert np.array_equal(res.x[0].values, ref.x[0].values)
        assert np.array_equal(res.fnorms, ref.fnorms)
        assert np.array_equal(res.xnorms, ref.xnorms)


def test_newton_state_and_context():
    "The state and the context in the hooks of Newton's method."

    class Recorder(fem.Plugin):
        def __init__(self):
            self.states = []
            self.log = []

        def before_newton(self, context, state):
            assert state.iteration is None
            assert state.x is not None
            assert state.fun is None  # not yet evaluated
            assert context.fun is not None
            assert context.update is not None
            assert context.check is not None
            assert context.items is not None
            self.states.append(state)
            self.log.append("before_newton")

        def before_iteration(self, context, state):
            assert state is self.states[0]
            assert state.alpha is None
            assert not state.updated
            assert state.fun is not None
            self.log.append(("before_iteration", state.iteration))

        def before_linear_solve(self, context, state):
            assert state.jac is not None
            self.x0 = state.x

        def after_linear_solve(self, context, state):
            assert state.x is self.x0
            assert state.dx is not None
            assert not state.updated

            # the context-callables evaluate the objective function without side
            # effects on the state variables
            x = context.update(state.x, state.dx)
            xnorm, fnorm, success = context.check(state.dx, x, context.fun(x))
            self.fnorm = fnorm

        def after_iteration(self, context, state):
            assert state is self.states[0]
            assert state.updated
            assert state.x is not self.x0
            assert np.isclose(state.fnorm, self.fnorm)
            self.log.append(("after_iteration", state.iteration))

        def after_newton(self, context, state):
            assert state.success
            self.log.append("after_newton")

    field, solid, loadcase = easy()
    recorder = Recorder()
    res = fem.newtonraphson(items=[solid], verbose=0, plugins=[recorder], **loadcase)

    assert res.success
    assert recorder.log[0] == "before_newton"
    assert recorder.log[-1] == "after_newton"
    assert recorder.log[1:-1:2] == [
        ("before_iteration", i) for i in range(res.iterations)
    ]
    assert recorder.log[2:-1:2] == [
        ("after_iteration", i) for i in range(res.iterations)
    ]


def test_newton_plugin_scales_increment():
    "A plugin may only replace the increment, the update is done by Newton's method."

    class Damping(fem.Plugin):
        def after_linear_solve(self, context, state):
            state.dx = 0.5 * state.dx
            state.alpha = 0.5

    increments = []

    def update(x, dx):
        increments.append(np.linalg.norm(dx))
        return x + dx

    field, solid, loadcase = easy()
    res = fem.newtonraphson(
        items=[solid],
        verbose=0,
        update=update,
        plugins=[Damping()],
        maxiter=64,
        **loadcase,
    )

    # linear convergence with a damping factor of 0.5
    assert res.success
    assert res.iterations > 8
    assert np.allclose(res.xnorms, increments)


if __name__ == "__main__":
    test_newton_plugins_unchanged()
    test_newton_state_and_context()
    test_newton_plugin_scales_increment()

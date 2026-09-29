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


def easy(umat=None):
    "A moderately stretched cube (plain Newton converges quadratically)."
    region = fem.RegionHexahedron(fem.Cube(n=4))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )
    if umat is None:
        umat = fem.NeoHooke(mu=1.0, bulk=2.0)
    solid = fem.SolidBody(umat=umat, field=field)
    return field, solid, loadcase


def hard(umat=None, planestrain=False):
    """A coarse, clamped mesh which is compressed by 70% in one step. The first Newton
    increment leads to inadmissible deformation gradients (det F <= 0)."""
    if planestrain:
        region = fem.RegionQuad(fem.Rectangle(n=3))
        field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        bulk = 20.0
    else:
        region = fem.RegionHexahedron(fem.Cube(n=3))
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        bulk = 5.0
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=-0.7, clamped=True, return_loadcase=True
    )
    if umat is None:
        umat = fem.NeoHooke(mu=1.0, bulk=bulk)
    solid = fem.SolidBody(umat=umat, field=field)
    return field, solid, loadcase


class NeoHookeTrace:
    """A Neo-Hookean material with a (fake) state variable, which stores the trace of
    the deformation gradient of the last evaluation of the gradient."""

    def __init__(self, bulk=5.0):
        self.material = fem.NeoHooke(mu=1.0, bulk=bulk)
        self.x = [np.eye(3), np.zeros(1)]

    def gradient(self, x):
        F = x[0]
        P = self.material.gradient([F, None])[0]
        return [P, np.trace(F)[None]]

    def hessian(self, x):
        return self.material.hessian([x[0], None])


def test_linesearch_none_is_default():
    results = []
    for kwargs in [{}, {"linesearch": None}, {"linesearch": False}]:
        field, solid, loadcase = easy()
        res = fem.newtonraphson(items=[solid], verbose=0, **loadcase, **kwargs)
        results.append(res)

    ref = results[0]
    assert ref.success
    assert ref.iterations == 4
    assert np.all(np.array(ref.alphas) == 1.0)

    for res in results[1:]:
        assert res.iterations == ref.iterations
        assert np.array_equal(res.fun, ref.fun)
        assert np.array_equal(res.x[0].values, ref.x[0].values)
        assert np.array_equal(res.fnorms, ref.fnorms)
        assert np.array_equal(res.xnorms, ref.xnorms)


def test_linesearch_full_steps():
    field, solid, loadcase = easy()
    ref = fem.newtonraphson(items=[solid], verbose=0, **loadcase)

    for linesearch in [True, fem.tools.LineSearch()]:
        field, solid, loadcase = easy()
        res = fem.newtonraphson(
            items=[solid], verbose=0, linesearch=linesearch, **loadcase
        )

        # the full Newton step is always accepted: quadratic convergence
        assert res.success
        assert res.iterations == ref.iterations
        assert np.all(np.array(res.alphas) == 1.0)
        assert np.allclose(res.fnorms, ref.fnorms, rtol=1e-10, atol=0)
        assert np.allclose(res.x[0].values, ref.x[0].values, rtol=1e-12, atol=0)


@pytest.mark.filterwarnings("ignore:Matrix is exactly singular")
@pytest.mark.parametrize("planestrain", [False, True])
def test_linesearch_inadmissible(planestrain):
    # plain Newton: the first increment leads to NaN
    field, solid, loadcase = hard(planestrain=planestrain)
    with pytest.raises(ValueError, match="NaN"):
        with np.errstate(all="ignore"):
            fem.newtonraphson(items=[solid], verbose=0, **loadcase)

    for linesearch in [
        fem.tools.LineSearch(),
        fem.tools.LineSearch(residual=False),
    ]:
        field, solid, loadcase = hard(planestrain=planestrain)
        with np.errstate(all="raise"):
            # no floating point warnings and no NaN values in the accepted trials
            res = fem.newtonraphson(
                items=[solid], verbose=0, linesearch=linesearch, **loadcase
            )

        assert res.success
        assert np.all(np.isfinite(res.fun))
        assert min(res.alphas) < 1
        assert len(res.alphas) == res.iterations

        F = solid.results.kinematics[0]
        assert np.all(fem.math.det(F) > 0)


def test_linesearch_max_halvings():
    with pytest.raises(ValueError, match="must not be negative"):
        fem.tools.LineSearch(max_halvings=-1)

    field, solid, loadcase = hard()
    x0 = field[0].values.copy()

    with pytest.raises(ValueError, match="Line search failed"):
        fem.newtonraphson(
            items=[solid],
            verbose=0,
            linesearch=fem.tools.LineSearch(max_halvings=0),
            **loadcase,
        )

    # the unknowns of the last accepted iteration are restored
    assert np.array_equal(field[0].values, x0)
    assert np.array_equal(solid.field[0].values, x0)


def test_linesearch_no_extra_assembly():
    """The residuals of the accepted trial are re-used, rejected trials (by the
    admissibility) are not assembled."""

    for problem, residual in [(easy, True), (hard, False)]:
        field, solid, loadcase = problem()
        umat = solid.umat

        counter = {"fun": 0}

        def fun(x, umat):
            counter["fun"] += 1
            return fem.tools._newton.fun(x, umat)

        res = fem.newtonraphson(
            x0=field,
            fun=fun,
            kwargs=dict(umat=umat),
            verbose=0,
            linesearch=fem.tools.LineSearch(residual=residual),
            **loadcase,
        )

        assert res.success
        assert counter["fun"] == 1 + res.iterations

        if problem is hard:
            assert min(res.alphas) < 1


def test_linesearch_statevars():
    "The temporary state variables belong to the accepted trial."

    field, solid, loadcase = hard(umat=NeoHookeTrace())

    def criterion(state):
        # reject all full steps after the assembly of the residuals
        assert state.assembled
        return state.alpha <= 0.5

    linesearch = fem.tools.LineSearch(criteria=[criterion])
    res = fem.newtonraphson(
        items=[solid], verbose=0, linesearch=linesearch, maxiter=64, **loadcase
    )

    assert res.success
    assert max(res.alphas) <= 0.5

    F = solid.field.extract()[0]
    assert np.allclose(solid.results.statevars[0], np.trace(F))
    assert np.allclose(solid.results._statevars[0], np.trace(F))


def test_linesearch_custom_criterion_and_update():
    field, solid, loadcase = easy()

    increments = []

    def update(x, dx):
        increments.append(np.linalg.norm(dx))
        return x + dx

    def half(state):
        assert isinstance(state, fem.tools.LineSearchState)
        return state.alpha <= 0.5

    res = fem.newtonraphson(
        items=[solid],
        verbose=0,
        update=update,
        linesearch=fem.tools.LineSearch(admissible=False, criteria=[half]),
        maxiter=64,
        **loadcase,
    )

    assert res.success
    assert np.all(np.array(res.alphas) == 0.5)

    # two calls of update (alpha=1 and alpha=0.5) per iteration
    increments = np.array(increments).reshape(-1, 2)
    assert np.allclose(increments[:, 1], increments[:, 0] / 2)

    # xnorms are based on the applied increments
    assert np.allclose(res.xnorms, increments[:, 1])

    # an in-place update is not supported
    def update_inplace(x, dx):
        x += dx
        return x

    field, solid, loadcase = easy()
    with pytest.raises(ValueError, match="in-place"):
        fem.newtonraphson(
            items=[solid],
            verbose=0,
            update=update_inplace,
            linesearch=True,
            **loadcase,
        )


@pytest.mark.filterwarnings("ignore:Matrix is exactly singular")
def test_linesearch_job():
    field, solid, loadcase = hard()
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    step = fem.Step(
        items=[solid],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.7, 0], num=1)},
        boundaries=boundaries,
    )

    with pytest.raises(ValueError, match="NaN"):
        with np.errstate(all="ignore"):
            fem.Job(steps=[step]).evaluate(x0=field, verbose=0)

    for linesearch in [True, fem.tools.LineSearch()]:
        field, solid, loadcase = hard()
        boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        step = fem.Step(
            items=[solid],
            ramp={boundaries["move"]: fem.math.linsteps([0, -0.7, 0], num=1)},
            boundaries=boundaries,
        )
        job = fem.Job(steps=[step])
        job.evaluate(x0=field, linesearch=linesearch, verbose=0)

        assert len(job.fnorms) == 3
        assert np.allclose(field[0].values, 0)


def test_linesearch_mixed():
    # third medium contact (mixed field formulation, plane strain)
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer(
        [fem.FieldPlaneStrain(region, dim=2), fem.Field(region, dim=9)],
        take=[0, 1, 1],
    )
    solid = fem.SolidBody(
        umat=fem.ThirdMediumContactMixed(
            material=fem.NeoHooke(mu=1, bulk=20), gamma=1e-5, alpha_r=1e-4, p_r=1e-2
        ),
        field=field,
        grad=[True, False, True],
    )
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    step = fem.Step(
        items=[solid],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.5], num=1)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(linesearch=True, verbose=0)
    assert np.all(fem.math.det(solid.results.kinematics[0]) > 0)

    # three-field variation and a nearly-incompressible solid body
    for mixed in [True, False]:
        if mixed:
            field = fem.FieldsMixed(fem.RegionHexahedron(fem.Cube(n=3)), n=3)
            umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
            solid = fem.SolidBody(umat, field)
        else:
            region = fem.RegionHexahedron(fem.Cube(n=3))
            field = fem.FieldContainer([fem.Field(region, dim=3)])
            umat = fem.NeoHooke(mu=1)
            solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

        boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        step = fem.Step(
            items=[solid],
            ramp={boundaries["move"]: fem.math.linsteps([0, -0.4], num=1)},
            boundaries=boundaries,
        )
        job = fem.Job(steps=[step])
        job.evaluate(linesearch=fem.tools.LineSearch(), verbose=0)
        assert np.isclose(job.fnorms[-1][-1], 0)


def test_linesearch_items_without_deformation_gradient():
    "Trusses and point loads are skipped by the admissibility criterion."

    mesh = fem.Mesh(
        points=[[0, 0], [1, 1], [2.0, 0]], cells=[[0, 1], [1, 2]], cell_type="line"
    )
    region = fem.RegionTruss(mesh)
    field = fem.Field(region, dim=2).as_container()
    boundaries = fem.BoundaryDict(fixed=fem.Boundary(field[0], fy=0))

    umat = fem.LinearElastic1D(E=np.ones(2))
    truss = fem.TrussBody(umat, field, area=np.ones(2))
    load = fem.PointLoad(field, [1])

    move = fem.math.linsteps([0, -0.1], num=5, axis=1, axes=2)
    step = fem.Step(items=[truss, load], ramp={load: move}, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate(linesearch=True, verbose=0)

    assert np.isclose(field[0].values[1, 1], -0.16302376)
    assert fem.tools._linesearch.deformation_gradients(field, [truss, load]) == []


def test_linesearch_pressure():
    "A solid body with a pressure boundary (which has a deformation gradient)."

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries, loadcase = fem.dof.uniaxial(
        field, move=0.2, clamped=True, return_loadcase=True
    )
    solid = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=2.0), field)

    regionp = fem.RegionHexahedronBoundary(mesh, only_surface=True)
    fieldp = fem.FieldContainer([fem.Field(regionp, dim=3)])
    pressure = fem.SolidBodyPressure(fieldp, pressure=0.1)

    Fs = fem.tools._linesearch.deformation_gradients(field, [solid, pressure])
    assert len(Fs) == 2

    res = fem.newtonraphson(
        items=[solid, pressure], verbose=0, linesearch=True, **loadcase
    )
    assert res.success


def test_linesearch_verbose(capsys):
    field, solid, loadcase = hard()
    fem.newtonraphson(
        items=[solid], verbose=2, linesearch=fem.tools.LineSearch(), **loadcase
    )
    out = capsys.readouterr().out
    assert "alpha=" in out

    field, solid, loadcase = easy()
    fem.newtonraphson(items=[solid], verbose=2, linesearch=True, **loadcase)
    out = capsys.readouterr().out
    assert "alpha=" not in out

    # progress bar with a postfix for reduced step lengths
    field, solid, loadcase = hard()
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    step = fem.Step(
        items=[solid],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.7, -0.75], num=1)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(x0=field, verbose=1, linesearch=True)


def test_linesearch_state():
    field, solid, loadcase = easy()
    f = fem.tools.fun([solid], field)

    calls = {"fun": 0}

    def fun(x):
        calls["fun"] += 1
        return fem.tools.fun([solid], x)

    dx = np.zeros(np.sum(field.fieldsizes))
    state = fem.tools.LineSearchState(
        x_old=field,
        x_trial=field + dx,
        f_old=f,
        alpha=1.0,
        dx=dx,
        items=[solid],
        fun=fun,
    )

    assert not state.assembled
    assert not state.converged  # no check function
    assert state.finite
    assert state.assembled
    assert calls["fun"] == 1

    # the residuals are cached
    state.f_trial
    assert calls["fun"] == 1

    # without items, the deformation gradient of the field container is used
    Fs = fem.tools._linesearch.deformation_gradients(field)
    assert len(Fs) == 1
    assert fem.tools._linesearch.deformation_gradients(np.zeros(3)) == []


if __name__ == "__main__":
    test_linesearch_none_is_default()
    test_linesearch_full_steps()
    test_linesearch_inadmissible(planestrain=False)
    test_linesearch_inadmissible(planestrain=True)
    test_linesearch_max_halvings()
    test_linesearch_no_extra_assembly()
    test_linesearch_statevars()
    test_linesearch_custom_criterion_and_update()
    test_linesearch_job()
    test_linesearch_mixed()
    test_linesearch_items_without_deformation_gradient()
    test_linesearch_pressure()
    test_linesearch_state()

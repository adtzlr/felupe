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

import contextlib
import os
import pathlib
import tempfile

import numpy as np
import pytest

import felupe as fem
from felupe.plugins._cutback import interpolate


class FailingNewton(fem.Plugin):
    """Raise an error in the Newton iteration ``iteration`` of the evaluations
    ``calls`` of the Newton-Raphson method (zero-based, counted during the whole job).
    For ``iteration > 0``, the items are already evaluated for the updated unknowns of
    the previous iterations, i.e. their states are modified by the failed attempt."""

    def __init__(self, calls, iteration=1, error=ValueError):
        self.calls = set(calls)
        self.iteration = iteration
        self.error = error
        self.count = -1
        self.raised = []

    def before_newton(self, context, state):
        self.count += 1

    def after_linear_solve(self, context, state):
        if self.count in self.calls and state.iteration == self.iteration:
            error = self.error(f"Injected failure in Newton evaluation {self.count}.")
            self.raised.append(error)
            raise error


class Recorder(fem.Plugin):
    "Record the hooks (without the hooks of the Newton iterations)."

    def __init__(self):
        self.hooks = []
        self.states = []

    def _record(self, hook, context, state):
        self.hooks.append(hook)
        self.states.append((context, state))

    def before_substep(self, context, state):
        self._record("before_substep", context, state)

    def before_newton(self, context, state):
        self._record("before_newton", context, state)

    def after_newton(self, context, state):
        self._record("after_newton", context, state)

    def after_failed_substep(self, context, state):
        self._record("after_failed_substep", context, state)

    def after_substep(self, context, state):
        self._record("after_substep", context, state)


def viscoelastic():
    "A (distortional) finite strain viscoelastic material with state variables."
    return fem.Hyperelastic(
        fem.constitution.finite_strain_viscoelastic,
        nstatevars=6,
        mu=1.0,
        eta=1.0,
        dtime=0.1,
    )


def initial_statevars(umat, region):
    """Return the initial state variables. The inelastic right Cauchy-Green deformation
    tensor of the viscoelastic material is initialized by the identity."""
    shape = (*umat.x[-1].shape, region.quadrature.npoints, region.mesh.ncells)
    statevars = np.zeros(shape)
    if shape[0] == 6:
        statevars[[0, 3, 5]] = 1.0
    return statevars


def cube(umat=None, nearly_incompressible=False, bulk=5.0):
    "A (coarse) cube with a uniaxial load case."
    region = fem.RegionHexahedron(fem.Cube(n=3))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

    if nearly_incompressible:
        if umat is None:
            umat = fem.NeoHooke(mu=1.0)
        statevars = initial_statevars(umat, region)
        solid = fem.SolidBodyNearlyIncompressible(
            umat, field, bulk=5000.0, statevars=statevars
        )
    else:
        if umat is None:
            umat = fem.NeoHooke(mu=1.0, bulk=bulk)
        statevars = initial_statevars(umat, region)
        solid = fem.SolidBody(umat, field, statevars=statevars)

    return field, boundaries, solid


class NewtonHistory(fem.Plugin):
    """Record the norms of the residuals of all iterations of all converged
    evaluations of the Newton-Raphson method (failed evaluations are not recorded)."""

    def __init__(self):
        self.fnorms = []
        self._fnorms = None

    def before_newton(self, context, state):
        self._fnorms = []

    def after_iteration(self, context, state):
        self._fnorms.append(state.fnorm)

    def after_newton(self, context, state):
        self.fnorms.append(self._fnorms)


def assert_same_history(history, reference):
    """The converged evaluations of the Newton-Raphson method (of the subdivided
    substeps) have the same iterations as the reference (with explicit substeps). This
    requires the exact restore of the unknowns and of the states of the items."""
    assert [len(fnorms) for fnorms in history.fnorms] == [
        len(fnorms) for fnorms in reference.fnorms
    ]
    for fnorms, fnorms_ref in zip(history.fnorms, reference.fnorms):
        assert np.allclose(fnorms, fnorms_ref, rtol=1e-8, atol=1e-13)


def run(ramp, plugins=None, **kwargs):
    """Evaluate a job of a cube with a ramp of the moved end face. The iterations of
    the Newton-Raphson method are recorded in ``job.history``."""
    field, boundaries, solid = cube(**kwargs)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )
    history = NewtonHistory()
    job = fem.Job(steps=[step], plugins=[*(plugins or []), history])
    job.evaluate(x0=field, verbose=0)
    job.history = history
    return field, solid, job


def refine(ramp, load_factors):
    """Return a ramp with the values of all increments of the substeps, i.e. explicit
    substeps instead of the subdivided substeps of a cutback."""
    values = []
    for substep, factors in enumerate(load_factors):
        for t in factors:
            values.append(interpolate({0: ramp[substep - 1]}, {0: ramp[substep]}, t)[0])
    return np.array(values)


def test_interpolate():
    values0 = {"a": 1.0, "b": np.array([0.0, 2.0])}
    values1 = {"a": 3.0, "b": np.array([4.0, 2.0])}

    values = interpolate(values0, values1, 0.25)
    assert np.isclose(values["a"], 1.5)
    assert np.allclose(values["b"], [1.0, 2.0])

    # the values of the substep are used for a load factor of one
    values = interpolate(values0, values1, 1.0)
    assert values["a"] is values1["a"]
    assert values["b"] is values1["b"]


def test_step_hooks():
    "Hooks of a substep, the context and the state."

    field, boundaries, solid = cube()
    ramp = fem.math.linsteps([0, 0.2, 0.4], num=1)
    move = boundaries["move"]
    step = fem.Step(items=[solid], ramp={move: ramp}, boundaries=boundaries)

    recorder = Recorder()
    failing = FailingNewton(calls=[2])

    class Retry(fem.Plugin):
        "Recover a failed substep by a second evaluation."

        def after_failed_substep(self, context, state):
            assert state.result is None
            state.result = context.solve(state.values)

    job = fem.Job(steps=[step], plugins=[failing, Retry(), recorder])
    job.evaluate(x0=field, verbose=0)

    assert recorder.hooks == [
        *["before_substep", "before_newton", "after_newton", "after_substep"] * 2,
        *["before_substep", "before_newton", "before_newton", "after_newton"],
        *["after_failed_substep", "after_substep"],
    ]

    # context and state of the substep hooks
    context, state = recorder.states[8]
    assert context.step is step
    assert context.items is step.items
    assert context.x0 is field
    assert callable(context.solve)
    assert isinstance(state, fem.SubstepState)
    assert isinstance(state, fem.JobState)
    assert state.stepnumber == 0
    assert state.substepnumber == 2
    assert state.time == 2
    assert state.values == {move: ramp[2]}

    # one state per substep, which is passed to all hooks of the substep
    for substep in range(3):
        hooks = recorder.hooks[4 * substep :]
        states = recorder.states[4 * substep :]
        before = states[hooks.index("before_substep")][1]
        after = states[hooks.index("after_substep")][1]
        assert after is before
        assert after.substepnumber == after.time == substep
        assert after.result.success

    context, state = recorder.states[12]
    assert state.error is failing.raised[0]
    assert isinstance(state.result, fem.tools.NewtonResult)
    assert state.result.success
    assert state.load_factors is None
    assert np.allclose(field[0].values, job.steps[0].items[0].field[0].values)

    # without a plugin, the error of the Newton-Raphson method is raised (unchanged)
    field, boundaries, solid = cube()
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )

    recorder = Recorder()
    failing = FailingNewton(calls=[1])

    with pytest.raises(ValueError, match="Injected failure") as excinfo:
        fem.Job(steps=[step], plugins=[failing, recorder]).evaluate(verbose=0)

    assert excinfo.value is failing.raised[0]
    assert recorder.hooks[-1] == "after_failed_substep"
    assert recorder.states[-1][1].error is failing.raised[0]


def test_step_unsuccessful_result():
    """A plugin may hand back an unsuccessful result of a failed substep. Then, the
    generation of the substeps of the step is stopped (without an error)."""

    class GiveUp(fem.Plugin):
        def after_failed_substep(self, context, state):
            state.result = fem.tools.NewtonResult(x=context.x0, success=False)

    field, boundaries, solid = cube()
    ramp = fem.math.linsteps([0, 0.2, 0.4], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )
    failing = FailingNewton(calls=[1])
    recorder = Recorder()
    job = fem.Job(steps=[step], plugins=[failing, GiveUp(), recorder])
    job.evaluate(verbose=0)

    # only the first substep is completed, the third substep is not evaluated
    assert len(job.fnorms) == 1
    assert recorder.hooks.count("before_substep") == 2
    assert recorder.hooks.count("after_substep") == 1


def test_step_generate_without_dispatcher():
    field, boundaries, solid = cube()
    ramp = fem.math.linsteps([0, 0.2], num=2)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )

    results = list(step.generate(x0=field, verbose=0))
    assert len(results) == 3
    assert all(res.success for res in results)

    states = list(step.generate_states(x0=field, verbose=0, stepnumber=4, time=7))
    assert [state.substepnumber for state in states] == [0, 1, 2]
    assert [state.time for state in states] == [7, 8, 9]
    assert all(state.stepnumber == 4 for state in states)

    # the unknowns are linked to the results, also for a top-level field container
    # which is not the field container of an item
    mesh_1 = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=3)
    mesh_2 = fem.Cube(a=(1, 0, 0), b=(2, 1, 1), n=3)
    field_1 = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh_1), dim=3)])
    field_2 = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh_2), dim=3)])
    x0 = fem.field.merge([field_1, field_2])
    solids = [
        fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), field_1),
        fem.SolidBody(fem.NeoHooke(mu=3.0, bulk=5.0), field_2),
    ]
    boundaries = fem.dof.uniaxial(x0, clamped=True, return_loadcase=False)
    step = fem.Step(
        items=solids, ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )
    for res in step.generate(x0=x0, verbose=0):
        assert x0[0].values is res.x[0].values

    assert np.all(x0[0].values[boundaries["move"].points, 0] == 0.2)

    # errors are raised
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: [np.nan]}, boundaries=boundaries
    )
    with pytest.raises(ValueError):
        with np.errstate(all="ignore"):
            list(step.generate(x0=field, verbose=0))


@pytest.mark.filterwarnings("ignore:Matrix is exactly singular")
def test_cutback_divergence():
    "A coarse cube is compressed by 70% in one substep (NaN values without cutback)."

    ramp = fem.math.linsteps([0, -0.7], num=1)

    with pytest.raises(ValueError, match="NaN"):
        with np.errstate(all="ignore"):
            run(ramp)

    cutback = fem.CutbackPlugin()

    with np.errstate(all="ignore"):
        field, solid, job = run(ramp, plugins=[cutback])

    assert cutback.load_factors == [[1.0], [0.5, 1.0]]
    assert cutback.cutbacks == [0, 1]

    # one result per substep
    assert len(job.fnorms) == 2
    assert len(job.timetrack) == 2

    # the result is identical to explicit substeps
    reference, _, job_ref = run(refine(ramp, cutback.load_factors))
    assert np.allclose(field[0].values, reference[0].values, rtol=1e-12, atol=1e-14)
    assert_same_history(job.history, job_ref.history)


@pytest.mark.parametrize("nearly_incompressible", [False, True])
def test_cutback_statevars(nearly_incompressible):
    """A viscoelastic material with state variables. The state variables are only
    updated for converged increments, i.e. the subdivided substep is identical to
    explicit substeps. The internal fields of a nearly-incompressible solid body are
    restored."""

    ramp = fem.math.linsteps([0, 0.3, 0.6], num=1)
    model = dict(umat=viscoelastic(), nearly_incompressible=nearly_incompressible)

    # fail the full substep and the second increment after one Newton iteration
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[2, 4], iteration=1)
    field, solid, job = run(ramp, plugins=[failing, cutback], **model)

    assert len(failing.raised) == 2
    assert cutback.load_factors == [[1.0], [1.0], [0.5, 0.75, 1.0]]
    assert cutback.cutbacks == [0, 0, 2]

    model = dict(umat=viscoelastic(), nearly_incompressible=nearly_incompressible)
    reference, solid_ref, job_ref = run(refine(ramp, cutback.load_factors), **model)

    assert_same_history(job.history, job_ref.history)
    assert np.allclose(field[0].values, reference[0].values, rtol=1e-12, atol=1e-14)
    assert np.allclose(
        solid.results.statevars, solid_ref.results.statevars, rtol=1e-12, atol=1e-14
    )

    # the state variables are path-dependent
    model = dict(umat=viscoelastic(), nearly_incompressible=nearly_incompressible)
    _, solid_direct, _ = run(ramp, **model)
    assert not np.allclose(solid.results.statevars, solid_direct.results.statevars)

    if nearly_incompressible:
        for name in ["u", "p", "J"]:
            value = getattr(solid.results.state, name)
            value_ref = getattr(solid_ref.results.state, name)
            assert np.allclose(value, value_ref, rtol=1e-12, atol=1e-14)


def test_cutback_plasticity():
    "Plasticity with a ramped (vector-valued) body force."

    def model():
        mesh = fem.Cube(b=(3, 1, 1), n=(4, 2, 2))
        region = fem.RegionHexahedron(mesh)
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        boundaries = fem.BoundaryDict(fixed=fem.dof.Boundary(field[0], fx=0))
        umat = fem.LinearElasticPlasticIsotropicHardening(
            E=2.1e5, nu=0.3, sy=355, K=1e3
        )
        solid = fem.SolidBody(umat, field)
        bodyforce = fem.SolidBodyForce(field)
        return field, boundaries, solid, bodyforce

    ramp = fem.math.linsteps([0, 400], num=4, axis=0, axes=3)

    field, boundaries, solid, bodyforce = model()
    step = fem.Step(
        items=[solid, bodyforce], ramp={bodyforce: ramp}, boundaries=boundaries
    )
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[3], iteration=1)
    history = NewtonHistory()
    fem.Job(steps=[step], plugins=[failing, cutback, history]).evaluate(verbose=0)

    assert len(failing.raised) == 1
    assert cutback.load_factors[3] == [0.5, 1.0]

    field_ref, boundaries, solid_ref, bodyforce = model()
    step = fem.Step(
        items=[solid_ref, bodyforce],
        ramp={bodyforce: refine(ramp, cutback.load_factors)},
        boundaries=boundaries,
    )
    history_ref = NewtonHistory()
    fem.Job(steps=[step], plugins=[history_ref]).evaluate(verbose=0)

    assert_same_history(history, history_ref)

    assert np.any(solid.results.statevars != 0)
    assert np.allclose(field[0].values, field_ref[0].values, rtol=1e-10, atol=1e-14)
    assert np.allclose(
        solid.results.statevars, solid_ref.results.statevars, rtol=1e-10, atol=1e-12
    )


def test_nearly_incompressible_restore():
    "The internal fields of a nearly-incompressible solid body are restored exactly."

    field, boundaries, solid = cube(nearly_incompressible=True)
    step = fem.Step(
        items=[solid],
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.2], num=2)},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(verbose=0)

    checkpoint = solid.checkpoint()
    F = solid.results.kinematics[0].copy()
    stress = solid.results.stress[0].copy()

    # evaluate the solid body for other (inadmissible) displacements
    field[0].values *= 5
    with np.errstate(all="ignore"):
        solid.assemble.vector(field)
        solid.assemble.matrix(field)

    assert not np.allclose(solid.results.state.J, checkpoint["results.state.J"])

    solid.restore(checkpoint)

    state = solid.results.state
    assert np.all(state.J == checkpoint["results.state.J"])
    assert np.all(state.p == checkpoint["results.state.p"])
    assert np.all(state.u == checkpoint["results.state.u"])
    assert np.allclose(solid.results.kinematics[0], F)
    assert np.allclose(state.F[0], F)
    assert np.allclose(solid.results.stress[0], stress)

    # re-initialize the internal fields by the restored field
    solid.restore(checkpoint, restore_state=False)
    assert np.allclose(state.J, state.volume() / solid.V)
    assert np.allclose(state.p, solid.bulk * (state.J - 1))


@pytest.mark.filterwarnings("ignore:Matrix is exactly singular")
def test_cutback_nearly_incompressible_divergence():
    "A nearly-incompressible solid body, compressed by 70% in one substep."

    ramp = fem.math.linsteps([0, -0.7], num=1)
    cutback = fem.CutbackPlugin()

    with np.errstate(all="ignore"):
        field, solid, job = run(ramp, plugins=[cutback], nearly_incompressible=True)

    assert min(cutback.cutbacks) == 0
    assert max(cutback.cutbacks) > 0

    reference, solid_ref, job_ref = run(
        refine(ramp, cutback.load_factors), nearly_incompressible=True
    )
    assert_same_history(job.history, job_ref.history)
    assert np.allclose(field[0].values, reference[0].values, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        solid.results.state.p, solid_ref.results.state.p, rtol=1e-10, atol=1e-10
    )


def test_cutback_growth_and_factor():
    ramp = fem.math.linsteps([0, 0.4], num=1)

    for growth, load_factors in [
        (1.0, [0.25, 0.5, 0.75, 1.0]),
        (2.0, [0.25, 0.75, 1.0]),
    ]:
        cutback = fem.CutbackPlugin(factor=0.25, growth=growth)
        failing = FailingNewton(calls=[1])
        field, solid, job = run(ramp, plugins=[failing, cutback])

        assert cutback.load_factors[-1] == load_factors
        assert cutback.cutbacks[-1] == 1

        reference, _, job_ref = run(refine(ramp, cutback.load_factors))
        assert np.allclose(field[0].values, reference[0].values)
        assert_same_history(job.history, job_ref.history)


def test_cutback_max_cutbacks():
    ramp = fem.math.linsteps([0, 0.4], num=1)

    # all attempts of the second substep fail
    cutback = fem.CutbackPlugin(max_cutbacks=2)
    failing = FailingNewton(calls=range(1, 10), iteration=1)

    with pytest.raises(ValueError, match="not recovered after 2 cutbacks") as excinfo:
        run(ramp, plugins=[failing, cutback])

    assert excinfo.value.__cause__ is failing.raised[-1]
    assert len(failing.raised) == 3
    assert cutback.load_factors == [[1.0], []]
    assert cutback.cutbacks == [0, 2]

    # no cutback at all
    cutback = fem.CutbackPlugin(max_cutbacks=0)
    failing = FailingNewton(calls=[1], iteration=1)

    with pytest.raises(ValueError, match="after 0 cutbacks") as excinfo:
        run(ramp, plugins=[failing, cutback])

    assert excinfo.value.__cause__ is failing.raised[0]

    # the first increment converges, the unknowns and the items are restored to the
    # last converged increment (the internal fields of a nearly-incompressible solid)
    field, boundaries, solid = cube(nearly_incompressible=True)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )
    cutback = fem.CutbackPlugin(max_cutbacks=3)
    failing = FailingNewton(calls=[1, *range(3, 10)], iteration=1)

    with pytest.raises(ValueError, match="last converged load factor 0.5"):
        fem.Job(steps=[step], plugins=[failing, cutback]).evaluate(verbose=0)

    assert cutback.load_factors[-1] == [0.5]
    assert cutback.cutbacks[-1] == 3

    reference, solid_ref, _ = run(
        refine(ramp, [[1.0], [0.5]]), nearly_incompressible=True
    )
    assert np.allclose(field[0].values, reference[0].values, rtol=1e-12, atol=1e-14)
    assert np.allclose(
        solid.field[0].values, reference[0].values, rtol=1e-12, atol=1e-14
    )
    for name in ["u", "p", "J"]:
        value = getattr(solid.results.state, name)
        value_ref = getattr(solid_ref.results.state, name)
        assert np.allclose(value, value_ref, rtol=1e-12, atol=1e-14)


def test_cutback_exceptions():
    ramp = fem.math.linsteps([0, 0.4], num=1)

    # errors of other types are raised without a cutback
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[1], error=TypeError)

    with pytest.raises(TypeError, match="Injected failure") as excinfo:
        run(ramp, plugins=[failing, cutback])

    assert excinfo.value is failing.raised[0]
    assert cutback.load_factors == [[1.0], []]

    # a single type
    cutback = fem.CutbackPlugin(exceptions=TypeError)
    failing = FailingNewton(calls=[1], error=TypeError)
    run(ramp, plugins=[failing, cutback])

    assert cutback.exceptions == (TypeError,)
    assert cutback.load_factors == [[1.0], [0.5, 1.0]]


def test_cutback_recovered_by_other_plugin():
    """A substep which is already recovered by a previous plugin is not modified by
    the cutback. Its values of the ramp are used for a later cutback."""

    class RetryOnce(fem.Plugin):
        "Recover the first failed substep by a second evaluation."

        def __init__(self):
            self.results = []

        def after_failed_substep(self, context, state):
            if len(self.results) == 0:
                state.result = context.solve(state.values)
                self.results.append(state.result)

    ramp = fem.math.linsteps([0, 0.2, 0.4], num=1)
    retry = RetryOnce()
    cutback = fem.CutbackPlugin()

    # count the restores of the cutback
    restores = []
    restore = cutback.restore

    def spy(context, checkpoint):
        restores.append(checkpoint)
        return restore(context, checkpoint)

    cutback.restore = spy

    # the second substep is recovered by the first plugin, the third substep (the
    # fourth evaluation of the Newton-Raphson method) by the cutback
    failing = FailingNewton(calls=[1, 3])
    recorder = Recorder()
    field, solid, job = run(ramp, plugins=[failing, retry, cutback, recorder])

    assert len(failing.raised) == 2
    assert len(retry.results) == 1

    # the result of the first plugin is used for the second substep
    failed = [
        state
        for hook, (context, state) in zip(recorder.hooks, recorder.states)
        if hook == "after_failed_substep"
    ]
    assert failed[0].substepnumber == 1
    assert failed[0].result is retry.results[0]
    assert failed[0].load_factors is None
    assert failed[1].substepnumber == 2
    assert failed[1].load_factors == [0.5, 1.0]

    # the second substep is neither restored nor subdivided by the cutback
    assert len(restores) == 1
    assert cutback.load_factors == [[1.0], [1.0], [0.5, 1.0]]
    assert cutback.cutbacks == [0, 0, 1]

    reference, _, _ = run(refine(ramp, cutback.load_factors))
    assert np.allclose(field[0].values, reference[0].values)


def test_cutback_unknown_values():
    # the values of the previous substep of the first substep are unknown
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[0])

    with pytest.raises(ValueError, match="can not be subdivided") as excinfo:
        run([0.4], plugins=[failing, cutback])

    assert excinfo.value.__cause__ is failing.raised[0]
    assert cutback.load_factors == [[]]

    # a step without a ramp
    field, boundaries, solid = cube()
    boundaries["move"].update(0.4)
    step = fem.Step(items=[solid], boundaries=boundaries)
    failing = FailingNewton(calls=[0])

    with pytest.raises(ValueError, match="the step has no ramp"):
        fem.Job(steps=[step], plugins=[failing, fem.CutbackPlugin()]).evaluate(
            verbose=0
        )


def test_cutback_multiple_steps_and_jobs():
    """The values of the ramp of the previous substep are taken from the previous
    step (and job)."""

    def steps(field, boundaries, solid):
        move = boundaries["move"]
        step_1 = fem.Step(
            items=[solid], ramp={move: np.array([0.0, 0.2])}, boundaries=boundaries
        )
        # the first substep of the second step is a jump
        step_2 = fem.Step(
            items=[solid], ramp={move: np.array([0.4, 0.5])}, boundaries=boundaries
        )
        return step_1, step_2

    field, boundaries, solid = cube(umat=viscoelastic())
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[2])
    job = fem.Job(steps=steps(field, boundaries, solid), plugins=[failing, cutback])
    job.evaluate(verbose=0)

    assert cutback.load_factors == [[1.0], [1.0], [0.5, 1.0], [1.0]]

    field_ref, boundaries, solid_ref = cube(umat=viscoelastic())
    move = boundaries["move"]
    ramp = np.array([0.0, 0.2, 0.3, 0.4, 0.5])
    step = fem.Step(items=[solid_ref], ramp={move: ramp}, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate(verbose=0)

    assert np.allclose(field[0].values, field_ref[0].values)
    assert np.allclose(solid.results.statevars, solid_ref.results.statevars)

    # a new job with the same plugin: the values of the previous job are known
    field, boundaries, solid = cube()
    move = boundaries["move"]
    cutback = fem.CutbackPlugin()
    step = fem.Step(items=[solid], ramp={move: [0.0, 0.2]}, boundaries=boundaries)
    fem.Job(steps=[step], plugins=[cutback]).evaluate(verbose=0)

    step = fem.Step(items=[solid], ramp={move: [0.4]}, boundaries=boundaries)
    failing = FailingNewton(calls=[0])
    fem.Job(steps=[step], plugins=[failing, cutback]).evaluate(verbose=0)

    assert cutback.load_factors == [[1.0], [1.0], [0.5, 1.0]]


@pytest.mark.filterwarnings("ignore:Matrix is exactly singular")
def test_cutback_linesearch():
    "A failed line search leads to a cutback."

    ramp = fem.math.linsteps([0, -0.7], num=1)

    linesearch = fem.LinesearchPlugin(max_halvings=0)
    with pytest.raises(ValueError, match="Line search failed"):
        with np.errstate(all="ignore"):
            run(ramp, plugins=[linesearch])

    linesearch = fem.LinesearchPlugin(max_halvings=0)
    cutback = fem.CutbackPlugin()
    with np.errstate(all="ignore"):
        field, solid, job = run(ramp, plugins=[linesearch, cutback])

    assert cutback.cutbacks[-1] > 0
    assert cutback.load_factors[-1][-1] == 1.0

    # one list of step lengths for each evaluation of the Newton-Raphson method
    assert len(linesearch.alphas) > len(job.fnorms)
    assert np.all(fem.math.det(solid.results.kinematics[0]) > 0)


def test_cutback_thermal():
    """A thermal transient analysis with radiation. The time is interpolated and the
    old time of the time step item is restored."""

    def model():
        mesh = fem.Rectangle(n=3)
        region = fem.RegionQuad(mesh)
        temperature = fem.Field(region, dim=1, values=20.0)
        field = fem.FieldContainer([temperature])

        region_top = fem.RegionQuadBoundary(mesh, mask=mesh.y == 1.0)
        field_top = fem.FieldContainer([fem.Field(region_top, dim=1)])

        boundaries = fem.BoundaryDict(
            left=fem.Boundary(temperature, fx=0, value=20.0),
        )
        solid = fem.thermal.SolidBodyThermal(
            field=field,
            mass_density=1.0,
            specific_heat_capacity=1.0,
            thermal_conductivity=1.0,
        )
        radiation = fem.thermal.SolidBodySurfaceRadiation(
            field=field_top, emissivity=0.8, temperature=10.0
        )
        time = fem.thermal.TimeStep([solid, radiation])
        return field, boundaries, solid, radiation, time

    def evaluate(table, plugins=None):
        field, boundaries, solid, radiation, time = model()
        ramp = {
            time: 0.1 * table,
            radiation["temperature"]: 10 + 500 * table,
            boundaries["left"]: 20 + 100 * table,
        }
        step = fem.Step(
            items=[time, solid, radiation], ramp=ramp, boundaries=boundaries
        )
        history = NewtonHistory()
        fem.Job(steps=[step], plugins=[*(plugins or []), history]).evaluate(verbose=0)
        return field, time, history

    table = fem.math.linsteps([0, 1, 2], num=1)
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[2, 4], iteration=1)
    field, time, history = evaluate(table, plugins=[failing, cutback])

    assert len(failing.raised) == 2
    assert cutback.load_factors[-1] == [0.5, 0.75, 1.0]
    assert np.isclose(time.time_old, 0.2)

    field_ref, time_ref, history_ref = evaluate(refine(table, cutback.load_factors))
    assert np.allclose(field[0].values, field_ref[0].values, rtol=1e-12)
    assert_same_history(history, history_ref)

    # checkpoint and restore of the time step item
    checkpoint = time.checkpoint()
    time.update(0.5)
    assert np.isclose(time.time_old, 0.5)
    time.restore(checkpoint)
    assert np.isclose(time.time_old, 0.2)

    # a failed first substep: the (uninitialized) temperature of the last time step is
    # restored and the substep can not be subdivided
    field, boundaries, solid, radiation, time = model()
    step = fem.Step(
        items=[time, solid, radiation],
        ramp={time: [0.1], boundaries["left"]: [30.0]},
        boundaries=boundaries,
    )
    assert solid.results.statevars.size == 0

    failing = FailingNewton(calls=[0], iteration=0)
    with pytest.raises(ValueError, match="can not be subdivided"):
        fem.Job(steps=[step], plugins=[failing, fem.CutbackPlugin()]).evaluate(
            verbose=0
        )

    assert solid.results.statevars.size == 0
    assert np.isclose(time.time_old, 0.0)
    assert np.allclose(field[0].values, 20.0)


def test_cutback_contact_friction():
    "The state of a frictional contact is restored."

    def model(stiffness_based):
        mesh = fem.Cube(n=3)
        mesh.add_points([2.0, 0.5, 0.5])
        region = fem.RegionHexahedron(mesh)
        displacement = fem.Field(region, dim=3)
        field = fem.FieldContainer([displacement])
        solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)
        contact = fem.ContactRigidPlane(
            field=field,
            points=np.arange(mesh.npoints)[mesh.x == 1],
            centerpoint=-1,
            normal=[-1.0, 0, 0],
            friction=0.3,
            multiplier=1e2,
            items=[solid] if stiffness_based else None,
        )
        boundaries = {
            "fixed": fem.Boundary(displacement, fx=0),
            "normal": fem.Boundary(displacement, fx=2, skip=(0, 1, 1)),
            "tangential": fem.Boundary(displacement, fx=2, skip=(1, 0, 1)),
            "other": fem.Boundary(displacement, fx=2, skip=(1, 1, 0)),
        }
        return field, solid, contact, boundaries

    def evaluate(normal, tangential, plugins=None, stiffness_based=False):
        field, solid, contact, boundaries = model(stiffness_based)
        ramp = {boundaries["normal"]: normal, boundaries["tangential"]: tangential}
        step = fem.Step([solid, contact], ramp=ramp, boundaries=boundaries)
        history = NewtonHistory()
        fem.Job(steps=[step], plugins=[*(plugins or []), history]).evaluate(verbose=0)
        contact.history = history
        return field, contact

    def assert_equal(field, contact, field_ref, contact_ref, rtol=1e-10, atol=1e-12):
        assert np.allclose(field[0].values, field_ref[0].values, rtol=rtol, atol=atol)
        for name in ["dx_ref", "active", "slip"]:
            assert np.allclose(
                getattr(contact.results, name),
                getattr(contact_ref.results, name),
                rtol=rtol,
                atol=atol,
            )

    normal = np.array([0, -1.2, -1.2, -1.2])

    # the Newton-Raphson method fails after one iteration (the contact state is
    # modified by the failed attempt)
    tangential = np.array([0, 0, 0.2, 0.4])
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[3], iteration=1)
    field, contact = evaluate(normal, tangential, plugins=[failing, cutback])

    assert len(failing.raised) == 1
    assert cutback.load_factors[-1] == [0.5, 1.0]
    assert np.all(contact.results.active)

    # the contact is sliding: the tangential reference gap vectors are moved
    assert np.any(np.abs(contact.results.dx_ref[:, 1]) > 0.1)

    field_ref, contact_ref = evaluate(
        refine(normal, cutback.load_factors),
        refine(tangential, cutback.load_factors),
    )
    assert_same_history(contact.history, contact_ref.history)
    assert_equal(field, contact, field_ref, contact_ref)

    # the Newton-Raphson method does not converge for a large tangential movement
    # with stiffness-based multipliers
    tangential = np.array([0, 0, 0.5, 1.0])

    with pytest.raises(ValueError, match="Maximum number of iterations"):
        evaluate(normal, tangential, stiffness_based=True)

    cutback = fem.CutbackPlugin()
    field, contact = evaluate(
        normal, tangential, plugins=[cutback], stiffness_based=True
    )
    assert max(cutback.cutbacks) > 0

    # the stiffness-based multipliers depend on the last assembled stiffness matrix
    # of the solid body. This matrix is re-assembled for the restored solid body.
    # Hence, the results are not identical to explicit substeps (but close).
    field_ref, contact_ref = evaluate(
        refine(normal, cutback.load_factors),
        refine(tangential, cutback.load_factors),
        stiffness_based=True,
    )
    assert_equal(field, contact, field_ref, contact_ref, rtol=1e-5, atol=1e-7)

    # checkpoint and restore of the contact
    checkpoint = contact.checkpoint()
    dx_ref = contact.results.dx_ref.copy()
    active = contact.results.active.copy()
    contact.results.dx_ref[:] = 0.0
    contact.results.active[:] = False
    contact.restore(checkpoint)
    assert np.allclose(contact.results.dx_ref, dx_ref)
    assert np.all(contact.results.active == active)
    assert contact.results.force is None


def test_cutback_items_without_checkpoint():
    """Only the unknowns are restored for items without the methods ``checkpoint()``
    and ``restore()``, e.g. a truss body and a point load (with a vector-valued
    ramp)."""

    def evaluate(table, plugins=None):
        mesh = fem.Mesh(
            points=[[0, 0], [1, 1], [2.0, 0]],
            cells=[[0, 1], [1, 2]],
            cell_type="line",
        )
        region = fem.RegionTruss(mesh)
        field = fem.Field(region, dim=2).as_container()
        boundaries = fem.BoundaryDict(fixed=fem.Boundary(field[0], fy=0))

        truss = fem.TrussBody(fem.LinearElastic1D(E=[1, 1]), field, area=[1, 1])
        load = fem.PointLoad(field, points=[1])
        step = fem.Step(
            items=[truss, load], ramp={load: table * -0.1}, boundaries=boundaries
        )
        history = NewtonHistory()
        fem.Job(steps=[step], plugins=[*(plugins or []), history]).evaluate(verbose=0)
        return field, truss, load, history

    table = fem.math.linsteps([0, 1], num=2, axis=1, axes=2)
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[2], iteration=1)
    field, truss, load, history = evaluate(table, plugins=[failing, cutback])

    assert len(failing.raised) == 1
    assert cutback._stateful(fem.Context(items=[truss, load])) == []
    assert cutback.load_factors == [[1.0], [1.0], [0.5, 1.0]]
    assert np.allclose(load.values, table[-1] * -0.1)

    field_ref, _, _, history_ref = evaluate(refine(table, cutback.load_factors))
    assert np.allclose(field[0].values, field_ref[0].values, rtol=1e-12, atol=1e-14)
    assert_same_history(history, history_ref)


def test_cutback_merged_fields():
    """Two solid bodies with merged fields, i.e. with a top-level field container as
    unknowns (which is not a field container of an item)."""

    def evaluate(ramp, plugins=None):
        mesh_1 = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=3)
        mesh_2 = fem.Cube(a=(1, 0, 0), b=(2, 1, 1), n=3)
        field_1 = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh_1), dim=3)])
        field_2 = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh_2), dim=3)])
        field = fem.field.merge([field_1, field_2])

        solid_1 = fem.SolidBody(fem.NeoHooke(mu=1.0, bulk=5.0), field_1)
        solid_2 = fem.SolidBody(fem.NeoHooke(mu=3.0, bulk=5.0), field_2)

        boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        step = fem.Step(
            items=[solid_1, solid_2],
            ramp={boundaries["move"]: ramp},
            boundaries=boundaries,
        )
        history = NewtonHistory()
        job = fem.Job(steps=[step], plugins=[*(plugins or []), history])
        job.evaluate(verbose=0)
        return field, solid_1, history

    ramp = fem.math.linsteps([0, 1.0], num=1)
    cutback = fem.CutbackPlugin(factor=0.25)
    failing = FailingNewton(calls=[1], iteration=1)
    field, solid, history = evaluate(ramp, plugins=[failing, cutback])

    assert solid.field.x0 is field
    assert cutback.load_factors[-1] == [0.25, 0.5, 0.75, 1.0]

    # the top-level field container holds the prescribed displacement of the end face
    assert np.isclose(field[0].values[:, 0].max(), 1.0)

    field_ref, _, history_ref = evaluate(refine(ramp, cutback.load_factors))
    assert_same_history(history, history_ref)
    assert np.allclose(field[0].values, field_ref[0].values, rtol=1e-12, atol=1e-14)


@contextlib.contextmanager
def working_directory(path):
    "Change the working directory (meshio writes the h5-file relative to it)."
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def test_cutback_xdmf(tmp_path):
    """A result file is written for all substeps, if the Newton-Raphson method of a
    substep reaches the maximum number of iterations and the substep is recovered."""

    meshio = pytest.importorskip("meshio")
    pytest.importorskip("h5py")

    field, boundaries, solid = cube()
    ramp = np.array([0.0, 1.0, 1.2])
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )
    cutback = fem.CutbackPlugin()
    job = fem.Job(steps=[step], plugins=[cutback])

    with working_directory(tmp_path):
        job.evaluate(verbose=0, filename="result.xdmf", maxiter=4)

        with meshio.xdmf.TimeSeriesReader("result.xdmf") as reader:
            reader.read_points_cells()
            num_steps = reader.num_steps
            data = [reader.read_data(k) for k in range(num_steps)]

    assert max(cutback.cutbacks) > 0
    assert job.timetrack == [0, 1, 2]
    assert num_steps == 3
    assert [time for time, point_data, cell_data in data] == [0, 1, 2]
    point_data = data[-1][1]

    # the displacements of the last substep
    assert np.allclose(point_data["Displacement"], field[0].values)


def test_cutback_characteristic_curve():
    "Other plugins are called once per substep."

    field, boundaries, solid = cube()
    move = boundaries["move"]
    step = fem.Step(
        items=[solid],
        ramp={move: fem.math.linsteps([0, 0.2, 0.4], num=1)},
        boundaries=boundaries,
    )
    cutback = fem.CutbackPlugin()
    failing = FailingNewton(calls=[2])
    curve = fem.CharacteristicCurvePlugin(boundary=move)
    job = fem.Job(steps=[step], plugins=[failing, cutback, curve])
    job.evaluate(verbose=0)

    assert cutback.load_factors[-1] == [0.5, 1.0]
    assert len(curve.x) == 3
    assert len(curve.y) == 3
    assert len(job.fnorms) == 3


def test_cutback_verbose(capsys):
    ramp = fem.math.linsteps([0, 0.4], num=1)

    failing = FailingNewton(calls=[1])
    field, boundaries, solid = cube()
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )
    job = fem.Job(steps=[step], plugins=[failing, fem.CutbackPlugin()])
    job.evaluate(verbose=2)

    out = capsys.readouterr().out
    assert "Substep 1/2 of Step 1/1 successful.\n" in out
    assert (
        "Substep 2/2 of Step 1/1 successful in 2 increments (load factors 0.5, 1)."
        in out
    )

    # progress bar with a postfix
    failing = FailingNewton(calls=[1])
    field, boundaries, solid = cube()
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: ramp}, boundaries=boundaries
    )
    job = fem.Job(steps=[step], plugins=[failing, fem.CutbackPlugin()])
    job.evaluate(verbose=1)


def test_cutback_invalid_parameters():
    for kwargs in [
        dict(factor=0.0),
        dict(factor=1.0),
        dict(max_cutbacks=-1),
        dict(growth=0.5),
    ]:
        with pytest.raises(ValueError):
            fem.CutbackPlugin(**kwargs)

    # the plugin is not a simple callable plugin
    cutback = fem.CutbackPlugin()
    job = fem.Job(steps=[], plugins=[cutback])
    assert not callable(cutback)
    assert cutback.after_substep in job.dispatcher.dispatcher["after_substep"]
    assert cutback.before_substep in job.dispatcher.dispatcher["before_substep"]


if __name__ == "__main__":
    test_interpolate()
    test_step_hooks()
    test_step_unsuccessful_result()
    test_step_generate_without_dispatcher()
    test_cutback_divergence()
    test_cutback_statevars(nearly_incompressible=False)
    test_cutback_statevars(nearly_incompressible=True)
    test_cutback_plasticity()
    test_nearly_incompressible_restore()
    test_cutback_nearly_incompressible_divergence()
    test_cutback_growth_and_factor()
    test_cutback_max_cutbacks()
    test_cutback_exceptions()
    test_cutback_recovered_by_other_plugin()
    test_cutback_unknown_values()
    test_cutback_multiple_steps_and_jobs()
    test_cutback_linesearch()
    test_cutback_thermal()
    test_cutback_contact_friction()
    with tempfile.TemporaryDirectory() as tmp:
        test_cutback_xdmf(tmp_path=pathlib.Path(tmp))
    test_cutback_characteristic_curve()
    test_cutback_invalid_parameters()

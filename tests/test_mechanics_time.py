# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import felupe as fem


def viscoelastic(dtime=None):
    """Return a time-dependent viscoelastic material (or a material with a constant time
    increment as material parameter)."""

    kwargs = {} if dtime is None else {"dtime": dtime}

    return fem.Hyperelastic(
        fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, nstatevars=6, **kwargs
    )


def model(Body="solid", dtime=None):
    "Return the field, the boundaries and a viscoelastic solid body of a cube."

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

    if Body == "solid":
        umat = viscoelastic(dtime) & fem.Volumetric(bulk=20.0)
        solid = fem.SolidBody(umat, field)

    elif Body == "nearly-incompressible":
        solid = fem.SolidBodyNearlyIncompressible(viscoelastic(dtime), field, bulk=20.0)

    return field, boundaries, solid


@pytest.mark.parametrize("Body", ["solid", "nearly-incompressible"])
def test_time_step_job(Body):
    """A time-dependent material with the time increment of a time step item equals
    the material with a constant time increment."""

    move = fem.math.linsteps([0, 0.3], num=3)
    dtime = 0.5

    # the time increment is an input of the material, updated by a time step item
    field, boundaries, solid = model(Body)
    assert solid.umat.time_dependent

    time = fem.TimeStep(items=[solid])
    step = fem.Step(
        items=[solid],
        ramp={boundaries["move"]: move, time: dtime * np.arange(len(move))},
        boundaries=boundaries,
    )
    fem.Job(steps=[step]).evaluate(verbose=0)
    assert np.isclose(solid.time_step, dtime)

    # the time increment is a constant material parameter
    # (the first substep is undeformed, i.e. the time increment of zero doesn't matter)
    field_old, boundaries_old, solid_old = model(Body, dtime=dtime)
    assert not solid_old.umat.time_dependent

    step = fem.Step(
        items=[solid_old],
        ramp={boundaries_old["move"]: move},
        boundaries=boundaries_old,
    )
    fem.Job(steps=[step]).evaluate(verbose=0)

    assert np.allclose(field[0].values, field_old[0].values, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        solid.results.statevars, solid_old.results.statevars, rtol=1e-10, atol=1e-12
    )

    # the deformation is viscoelastic
    assert not np.allclose(solid.results.statevars, 0.0)


def test_time_step_cutback():
    """The time increments of the increments of a subdivided substep are interpolated,
    i.e. a cutback equals the job with the substeps of the converged increments."""

    move = np.array([0.0, 1.0])
    times = np.array([0.0, 2.0])

    field, boundaries, solid = model()
    time = fem.TimeStep(items=[solid])
    step = fem.Step(
        items=[solid],
        ramp={boundaries["move"]: move, time: times},
        boundaries=boundaries,
    )
    cutback = fem.CutbackPlugin()
    fem.Job(steps=[step], plugins=[cutback]).evaluate(verbose=0, maxiter=6)

    # the second substep is subdivided
    load_factors = cutback.load_factors[1]
    assert len(load_factors) > 1

    # the reference job with the substeps of the converged increments
    t = np.array([0.0, *load_factors])
    field_ref, boundaries_ref, solid_ref = model()
    time_ref = fem.TimeStep(items=[solid_ref])
    step = fem.Step(
        items=[solid_ref],
        ramp={boundaries_ref["move"]: t * move[1], time_ref: t * times[1]},
        boundaries=boundaries_ref,
    )
    fem.Job(steps=[step]).evaluate(verbose=0)

    assert np.allclose(field[0].values, field_ref[0].values, rtol=1e-8, atol=1e-10)
    assert np.allclose(
        solid.results.statevars, solid_ref.results.statevars, rtol=1e-8, atol=1e-10
    )

    # with a constant time increment, each increment takes the full time increment
    field_old, boundaries_old, solid_old = model(dtime=times[1])
    step = fem.Step(
        items=[solid_old],
        ramp={boundaries_old["move"]: move},
        boundaries=boundaries_old,
    )
    cutback = fem.CutbackPlugin()
    fem.Job(steps=[step], plugins=[cutback]).evaluate(verbose=0, maxiter=6)

    assert cutback.load_factors[1] == load_factors
    assert not np.allclose(solid_old.results.statevars, solid_ref.results.statevars)


def test_time_step_relaxation():
    "The stress relaxes in a step in which only the time is ramped."

    field, boundaries, solid = model()
    time = fem.TimeStep(items=[solid])

    stress = []

    def record(context, state):
        stress.append(solid.results.stress[0][0, 0].mean())

    # a fast deformation followed by a hold step
    loading = fem.Step(
        items=[solid],
        ramp={
            boundaries["move"]: fem.math.linsteps([0, 0.3], num=3),
            time: fem.math.linsteps([0, 0.03], num=3),
        },
        boundaries=boundaries,
    )
    holding = fem.Step(
        items=[solid],
        ramp={time: fem.math.linsteps([0.03, 3.03], num=5)},
        boundaries=boundaries,
    )
    fem.Job(steps=[loading, holding], plugins=[record]).evaluate(verbose=0)

    # the displacements are kept constant in the hold step
    assert np.isclose(boundaries["move"].value, 0.3)

    # the stress of the hold step (its first substep has a time increment of zero)
    relaxation = np.array(stress[4:])
    assert relaxation[0] > 0
    assert np.all(np.diff(relaxation) < 0)

    # the volumetric part of the stress does not relax
    assert relaxation[-1] < 0.75 * relaxation[0]


def test_time_step_errors_and_options():
    "The time increment is required, kept on revolve and may be given explicitly."

    field, boundaries, solid = model()

    with pytest.raises(ValueError, match="TimeStep"):
        solid.assemble.vector(field)

    with pytest.raises(ValueError, match="TimeStep"):
        solid.assemble.matrix(field)

    # an explicitly given time increment is kept
    solid.time_step = 0.1
    r1 = solid.assemble.vector(field, kwargs={"dtime": 0.5}).toarray()
    solid.time_step = 0.5
    r2 = solid.assemble.vector(field).toarray()
    assert np.allclose(r1, r2)

    # the time increment is passed to a solid body by its argument
    for Body in [fem.SolidBody, fem.SolidBodyNearlyIncompressible]:
        kwargs = {} if Body is fem.SolidBody else {"bulk": 20.0}
        solid = Body(viscoelastic(), field, time_step=0.5, **kwargs)
        assert solid.time_step == 0.5
        solid.assemble.vector(field)
        solid.assemble.matrix(field)

    # materials which are not time-dependent don't take the time increment
    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field, time_step=0.5)
    solid.assemble.vector(field)
    solid.assemble.matrix(field)

    # the time increment is kept on revolve
    region = fem.RegionQuad(fem.Rectangle(n=3))
    axi = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])

    for Body, kwargs in [
        (fem.SolidBody, {}),
        (fem.SolidBodyNearlyIncompressible, {"bulk": 20.0}),
    ]:
        solid = Body(fem.NeoHooke(mu=1.0), axi, time_step=0.5, **kwargs)
        assert solid.revolve(n=3, phi=90).time_step == 0.5


def test_time_step_external_and_truss():
    "Materials of external packages and of trusses take the time increment."

    class External:
        "A time-dependent material which is not derived from ConstitutiveMaterial."

        time_dependent = True

        def __init__(self, x):
            self.x = x
            self.dtimes = []

        def gradient(self, x, dtime):
            self.dtimes.append(dtime)
            return [np.zeros_like(x[0]), x[-1]]

        def hessian(self, x, dtime):
            self.dtimes.append(dtime)
            shape = (*x[0].shape[:-2], *x[0].shape[:-2], *x[0].shape[-2:])
            return [np.zeros(shape)]

    field, boundaries, solid = model()

    umat = External([np.eye(3), np.zeros(0)])
    solid = fem.SolidBody(umat, field, time_step=0.25)
    solid.assemble.vector(field)
    solid.assemble.matrix(field)
    assert umat.dtimes == [0.25, 0.25]

    # a truss body with a time-dependent material of the stretch
    mesh = fem.mesh.Line(n=3)
    mesh.update(points=np.pad(mesh.points, ((0, 0), (0, 2))))
    truss = fem.FieldContainer([fem.Field(fem.RegionTruss(mesh), dim=3)])

    umat = External([np.ones(1), np.zeros(0)])
    body = fem.TrussBody(umat, truss, area=1.0, time_step=0.75)
    body.assemble.vector(truss)
    body.assemble.matrix(truss)
    assert len(umat.dtimes) > 0
    assert set(umat.dtimes) == {0.75}

    body.time_step = None
    with pytest.raises(ValueError, match="TrussBody is time-dependent"):
        body.assemble.vector(truss)


if __name__ == "__main__":
    for Body in ["solid", "nearly-incompressible"]:
        test_time_step_job(Body)

    test_time_step_cutback()
    test_time_step_relaxation()
    test_time_step_errors_and_options()
    test_time_step_external_and_truss()

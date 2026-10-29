# SPDX-License-Identifier: GPL-3.0-or-later

import contextlib
import io
import os
import pathlib
import tempfile
import warnings

import numpy as np
import pytest

import felupe as fem


def pre():
    mesh = fem.Rectangle(n=2)
    region = fem.RegionQuad(mesh)
    field = fem.FieldsMixed(region, n=3, axisymmetric=True)

    umat = fem.ThreeFieldVariation(fem.NeoHooke(1, 5000))
    body = fem.SolidBody(umat, field)
    boundaries = fem.dof.uniaxial(field, return_loadcase=False)

    points = mesh.points[:, 0] == 1
    load = fem.PointLoad(field, points)
    gravity = fem.SolidBodyForce(field, [0, 0, 0], 0)

    region2 = fem.RegionQuadBoundary(mesh, mask=points, ensure_3d=True)
    field2 = fem.FieldContainer([fem.FieldAxisymmetric(region2, dim=2)])
    pressure = fem.SolidBodyPressure(field2, pressure=0.0)

    step = fem.Step(
        items=[body, load, gravity, pressure],
        ramp={
            boundaries["move"]: fem.math.linsteps([0, 1], num=10),
            load: np.zeros((11, 2)),
            pressure: np.zeros(11),
            gravity: np.zeros((11, 3)),
        },
        boundaries=boundaries,
    )

    return field, step


def test_job():
    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()
    field, step = pre()
    substeps = []
    job = fem.Job(steps=[step], plugins=[lambda context, state: substeps.append(state)])
    job.evaluate(
        parallel=True,
        kwargs={"parallel": False},
        verbose=0,
    )
    assert [state.substepnumber for state in substeps] == list(range(11))


def test_job_xdmf(tmp_path):
    field, step = pre()

    job = fem.Job(steps=[step])
    job.evaluate()

    field, step = pre()
    job = fem.Job(steps=[step])

    with contextlib.chdir(tmp_path):
        job.evaluate(filename="result.xdmf", parallel=True, verbose=2)

    assert (tmp_path / "result.xdmf").exists()
    assert (tmp_path / "result.h5").exists()


def test_job_xdmf_time(tmp_path):
    "The (absolute) time of a time step item is the time of the XDMF result file."

    meshio = pytest.importorskip("meshio")

    def job(preload=False, plugin=None):
        """A job with a loading and a holding step, optionally with a preload step and
        a plugin which takes the time step item."""
        region = fem.RegionHexahedron(fem.Cube(n=2))
        field = fem.FieldContainer([fem.Field(region, dim=3)])
        boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
        time = fem.TimeStep(items=[solid])

        move = boundaries["move"]
        steps = [
            fem.Step(
                [solid],
                ramp={move: [0.1, 0.2], time: [0.5, 2.0]},
                boundaries=boundaries,
            ),
            fem.Step([solid], ramp={time: [3.0, 4.5]}, boundaries=boundaries),
        ]

        if preload:
            # a preload step without a ramp of the time, i.e. all substeps at t=0
            ramp = {move: [0.0, 0.05, 0.1]}
            steps.insert(0, fem.Step([solid], ramp=ramp, boundaries=boundaries))

        plugins = [] if plugin is None else [plugin(time)]
        return fem.Job(steps=steps, plugins=plugins), time

    def times(filename):
        "Return the times of a XDMF result file."
        with meshio.xdmf.TimeSeriesReader(filename) as reader:
            reader.read_points_cells()
            return [float(reader.read_data(k)[0]) for k in range(reader.num_steps)]

    with contextlib.chdir(tmp_path):
        # the times of the time step item (increasing, i.e. without a warning)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*XDMF result file")
            simulation, time = job()
            simulation.evaluate(filename="time.xdmf", time=time, verbose=0)

        assert np.allclose(times("time.xdmf"), [0.5, 2.0, 3.0, 4.5])

        # the indices of the substeps (default)
        simulation, time = job()
        simulation.evaluate(filename="index.xdmf", verbose=0)

        assert np.allclose(times("index.xdmf"), [0, 1, 2, 3])
        assert simulation.timetrack == [0, 1, 2, 3]

        # the plugin (with the default keyword-arguments)
        def writer(time):
            return fem.XDMFWriterPlugin(filename="plugin.xdmf", time=time)

        simulation, time = job(plugin=writer)
        simulation.evaluate(verbose=0)

        assert np.allclose(times("plugin.xdmf"), [0.5, 2.0, 3.0, 4.5])

        # repeated times are written with a warning (once)
        simulation, time = job(preload=True)
        with pytest.warns(UserWarning, match="not greater than the time") as record:
            simulation.evaluate(filename="preload.xdmf", time=time, verbose=0)

        warned = [w for w in record if "XDMF result file" in str(w.message)]
        assert len(warned) == 1
        assert np.allclose(times("preload.xdmf"), [0, 0, 0, 0.5, 2.0, 3.0, 4.5])


def test_job_xdmf_global_field(tmp_path):
    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()

    field, step = pre()
    job = fem.Job(steps=[step])

    with contextlib.chdir(tmp_path):
        job.evaluate(filename="result.xdmf", x0=field, tqdm="auto")

    assert (tmp_path / "result.xdmf").exists()


def test_job_xdmf_vertex(tmp_path):

    import felupe as fem

    meshes = [
        fem.Cube(n=3),
        fem.Cube(n=3).translate(1, axis=0),
    ]
    container = fem.MeshContainer(meshes, merge=True)
    field = fem.Field.from_mesh_container(container).as_container()

    regions = [
        fem.RegionHexahedron(container.meshes[0]),
        fem.RegionHexahedron(container.meshes[1]),
    ]
    fields = [
        fem.FieldContainer([fem.Field(regions[0], dim=3)]),
        fem.FieldContainer([fem.Field(regions[1], dim=3)]),
    ]

    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    umat = fem.LinearElasticLargeStrain(E=2.1e5, nu=0.3)
    solids = [
        fem.SolidBody(umat=umat, field=fields[0]),
        fem.SolidBody(umat=umat, field=fields[1]),
    ]

    move = fem.math.linsteps([0, 1], num=5)
    ramp = {boundaries["move"]: move}
    step = fem.Step(items=solids, ramp=ramp, boundaries=boundaries)

    job = fem.Job(steps=[step])

    with contextlib.chdir(tmp_path), pytest.warns(UserWarning):
        job.evaluate(x0=field, filename="result.xdmf", mesh=container.meshes[0])

    assert (tmp_path / "result.xdmf").exists()


def test_curve():
    field, step = pre()

    substeps = []
    curve = fem.CharacteristicCurve(
        steps=[step],
        boundary=step.boundaries["move"],
        plugins=[lambda context, state: substeps.append(state)],
    )

    with pytest.raises(ValueError):
        curve.plot()

    os.environ["FELUPE_VERBOSE"] = "true"

    curve.evaluate()
    curve.plot(xaxis=0, yaxis=0)
    curve.plot(x=np.zeros((10, 2)), y=np.ones((10, 2)), xaxis=0, yaxis=0)

    assert curve.res is not None
    assert curve.x is not None
    assert curve.y is not None

    stretch = 1 + np.array(curve.x)[:, 0]
    area = 1**2 * np.pi
    force = (stretch - 1 / stretch**2) * area

    os.environ.pop("FELUPE_VERBOSE")

    assert np.allclose(np.array(curve.y)[:, 0], force, rtol=0.01)
    assert len(substeps) == len(curve.x)


def test_curve2():
    field, step = pre()

    curve = fem.CharacteristicCurve(steps=[step], boundary=step.boundaries["move"])
    curve.evaluate()
    curve.plot(xaxis=0, yaxis=0)

    stretch = 1 + np.array(curve.x)[:, 0]
    area = 1**2 * np.pi
    force = (stretch - 1 / stretch**2) * area

    assert np.allclose(np.array(curve.y)[:, 0], force, rtol=0.01)


def test_curve_custom_items():
    field, step = pre()

    curve = fem.CharacteristicCurve(
        steps=[step], items=step.items, boundary=step.boundaries["move"]
    )
    curve.evaluate()
    fig, ax = curve.plot(
        xaxis=0, yaxis=0, gradient=True, swapaxes=True, xlabel="x", ylabel="y"
    )
    curve.plot(x=np.zeros((10, 2)), y=np.ones((10, 2)), xaxis=0, yaxis=0, ax=ax)

    stretch = 1 + np.array(curve.x)[:, 0]
    area = 1**2 * np.pi
    force = (stretch - 1 / stretch**2) * area

    assert np.allclose(np.array(curve.y)[:, 0], force, rtol=0.01)


def test_empty():
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldsMixed(region, n=1)

    umat = fem.NeoHooke(mu=1, bulk=5000)
    solid = fem.SolidBody(umat, field)

    step = fem.Step(items=[solid], ramp=None, boundaries=None)
    assert step.boundaries == {}

    job = fem.Job(steps=[step])

    with pytest.raises(ValueError):
        job.evaluate(tqdm="my_fancy_backend")

    boundaries = fem.dof.symmetry(field[0])
    step = fem.Step(items=[solid], ramp=None, boundaries=boundaries)
    job = fem.Job(steps=[step])

    job.evaluate(tqdm="notebook")


def test_noramp():
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldsMixed(region, n=1)

    umat = fem.LinearElastic(E=1, nu=0.3)
    solid = fem.SolidBody(umat, field)
    bounds = fem.dof.uniaxial(field, return_loadcase=False)

    step = fem.Step(items=[solid], ramp=None, boundaries=bounds)
    job = fem.Job(steps=[step])
    job.evaluate()


@pytest.mark.filterwarnings("ignore:Matrix is exactly singular")
def test_job_after_job_on_error(tmp_path):
    """The hook ``after_job`` is also triggered if the evaluation of a job fails, e.g.
    to close the result file. The error is raised after all plugins are called."""

    meshio = pytest.importorskip("meshio")
    pytest.importorskip("h5py")

    class Recorder(fem.Plugin):
        def __init__(self):
            self.states = []

        def after_job(self, context, state):
            self.states.append(state)

    region = fem.RegionHexahedron(fem.Cube(n=3))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)

    # the second substep fails (NaN values)
    move = fem.math.linsteps([0, -0.7], num=1)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    recorder = Recorder()
    job = fem.Job(steps=[step], plugins=[recorder])

    # meshio writes the h5-file relative to the working directory
    with contextlib.chdir(tmp_path):
        with pytest.raises(ValueError, match="NaN") as excinfo:
            with np.errstate(all="ignore"):
                job.evaluate(filename="result.xdmf", verbose=0)

        # the result file is closed and holds the completed substep
        with meshio.xdmf.TimeSeriesReader("result.xdmf") as reader:
            reader.read_points_cells()
            num_steps = reader.num_steps

    assert num_steps == 1
    assert len(recorder.states) == 1
    assert isinstance(recorder.states[0], fem.JobState)
    assert recorder.states[0].error is excinfo.value

    # without an error (a new model, the solid body is modified by the failed job)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
    recorder = Recorder()
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: [0.0, 0.1]}, boundaries=boundaries
    )
    fem.Job(steps=[step], plugins=[recorder]).evaluate(verbose=0)

    assert len(recorder.states) == 1
    assert recorder.states[0].error is None


def test_job_repeated_evaluate(tmp_path):
    """Repeated evaluations of a job don't register the plugins multiple times, see
    https://github.com/adtzlr/felupe/issues/1124."""

    pytest.importorskip("meshio")
    pytest.importorskip("h5py")

    class Recorder(fem.Plugin):
        def __init__(self):
            self.hooks = []

        def before_job(self, context, state):
            self.hooks.append("before_job")

        def after_substep(self, context, state):
            self.hooks.append("after_substep")

        def after_job(self, context, state):
            self.hooks.append("after_job")

    region = fem.RegionHexahedron(fem.Cube(n=2))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
    move = fem.math.linsteps([0, 0.1], num=2)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    recorder = Recorder()
    substeps = []
    plugins = [recorder, lambda context, state: substeps.append(state.substepnumber)]
    job = fem.Job(steps=[step], plugins=plugins)

    hooks = ["before_job", *["after_substep"] * step.nsubsteps, "after_job"]

    for _ in range(3):
        recorder.hooks.clear()
        substeps.clear()

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            job.evaluate(verbose=2)

        # the plugins of the job are called once per hook in each evaluation
        assert recorder.hooks == hooks
        assert substeps == list(range(step.nsubsteps))

        # the built-in progress plugin is not added to the dispatcher of the job
        assert job.dispatcher.plugins == plugins

        # one header is printed by one progress plugin in each evaluation
        assert stdout.getvalue().count("Run Job") == 1

    with contextlib.chdir(tmp_path):
        job.evaluate(filename="first.xdmf", verbose=0)
        assert (tmp_path / "first.xdmf").exists()

        for path in tmp_path.iterdir():
            path.unlink()

        # the writer of the first result file is not re-used (the file is not
        # overwritten by the second evaluation)
        job.evaluate(filename="second.xdmf", verbose=0)

    assert (tmp_path / "second.xdmf").exists()
    assert not (tmp_path / "first.xdmf").exists()
    assert job.dispatcher.plugins == plugins


def test_callback_removed():
    "The removed callbacks and their keyword-arguments raise a TypeError."

    field, step = pre()

    def callback(stepnumber, substepnumber, substep, **kwargs):
        pass

    with pytest.raises(TypeError):
        fem.Job(steps=[step], callback=callback)

    with pytest.raises(TypeError):
        fem.Job([step], callback)  # plugins is a keyword-only argument

    with pytest.raises(TypeError):
        fem.Job(steps=[step], plugin=[])  # e.g. a typo of plugins

    with pytest.raises(TypeError):
        fem.CharacteristicCurve(
            steps=[step], boundary=step.boundaries["move"], callback=callback
        )

    # the callback of newtonraphson() was passed by Job.evaluate()
    with pytest.raises(TypeError):
        fem.Job(steps=[step]).evaluate(callback=lambda *args: None, verbose=0)

    field, step = pre()
    _, loadcase = fem.dof.uniaxial(field, return_loadcase=True)

    for kwargs in [{"callback": print}, {"progress_bar": 1}, {"callback_kwargs": {}}]:
        with pytest.raises(TypeError):
            fem.newtonraphson(items=step.items, **loadcase, verbose=0, **kwargs)


def test_callback_plugins():
    "The plugins of the notes of Job and newtonraphson(), which replace callbacks."

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field)
    move = fem.math.linsteps([0, 0.2], num=2)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    substeps, iterations = [], []

    def callback(stepnumber, substepnumber, substep, **kwargs):
        substeps.append((stepnumber, substepnumber, substep, kwargs))

    def newton_callback(dx, x, iteration, xnorm, fnorm, success):
        iterations.append((iteration, fnorm, success))

    class NewtonCallback(fem.Plugin):
        def after_iteration(self, context, state):
            newton_callback(
                state.dx,
                state.x,
                state.iteration,
                state.xnorm,
                state.fnorm,
                state.success,
            )

    plugins = [
        lambda context, state: callback(
            state.stepnumber, state.substepnumber, context.substep, a=1
        ),
        NewtonCallback(),
    ]
    fem.Job(steps=[step], plugins=plugins).evaluate(verbose=0)

    assert [(i, j, kwargs) for i, j, _, kwargs in substeps] == [
        (0, j, {"a": 1}) for j in range(3)
    ]

    # the iterations of all substeps, the last one of each substep has converged
    results = [substep for _, _, substep, _ in substeps]
    assert len(iterations) == sum(res.iterations for res in results)
    converged = [(fnorm, success) for _, fnorm, success in iterations if success]
    assert converged == [(res.fnorms[-1], True) for res in results]


if __name__ == "__main__":
    test_job()

    for test in [
        test_job_xdmf,
        test_job_xdmf_time,
        test_job_xdmf_global_field,
        test_job_xdmf_vertex,
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            test(tmp_path=pathlib.Path(tmp))

    test_curve()
    test_curve2()
    test_curve_custom_items()
    test_empty()
    test_noramp()
    test_callback_removed()
    test_callback_plugins()

    with tempfile.TemporaryDirectory() as tmp:
        test_job_after_job_on_error(tmp_path=pathlib.Path(tmp))

    with tempfile.TemporaryDirectory() as tmp:
        test_job_repeated_evaluate(tmp_path=pathlib.Path(tmp))

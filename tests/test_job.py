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
import io
import os
import pathlib
import tempfile

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


def weather(i, j, res, outside):
    assert outside == "rainy"


def test_job():
    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()
    field, step = pre()
    job = fem.Job(steps=[step], callback=weather, outside="rainy")
    job.evaluate(
        parallel=True,
        kwargs={"parallel": False},
        verbose=0,
    )


def test_job_xdmf():
    field, step = pre()

    job = fem.Job(steps=[step])
    job.evaluate()

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate(filename="result.xdmf", parallel=True, verbose=2)


def test_job_xdmf_global_field():
    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate(filename="result.xdmf", x0=field, tqdm="auto")


def test_job_xdmf_vertex():

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

    with pytest.warns(UserWarning):
        job.evaluate(x0=field, filename="result.xdmf", mesh=container.meshes[0])


def test_curve():
    field, step = pre()

    curve = fem.CharacteristicCurve(
        steps=[step],
        boundary=step.boundaries["move"],
        callback=weather,
        outside="rainy",
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
    job = fem.Job(steps=[step])

    with pytest.raises(ValueError):
        job.evaluate(tqdm="my_fancy_backend")

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


@contextlib.contextmanager
def working_directory(path):
    "Change the working directory (meshio writes the h5-file relative to it)."
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


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

    with working_directory(tmp_path):
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

    with working_directory(tmp_path):
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


if __name__ == "__main__":
    test_job()
    test_job_xdmf()
    test_job_xdmf_global_field()
    test_job_xdmf_vertex()
    test_curve()
    test_curve2()
    test_curve_custom_items()
    test_empty()
    test_noramp()

    with tempfile.TemporaryDirectory() as tmp:
        test_job_after_job_on_error(tmp_path=pathlib.Path(tmp))

    with tempfile.TemporaryDirectory() as tmp:
        test_job_repeated_evaluate(tmp_path=pathlib.Path(tmp))

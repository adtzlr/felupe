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
import os

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
        job.evaluate(x0=field, filename="result.xdmf")


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

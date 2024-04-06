# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 00:27:38 2022

@author: z0039mte
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
    bounds, loadcase = fem.dof.uniaxial(field)

    points = mesh.points[:, 0] == 1
    load = fem.PointLoad(field, points)
    gravity = fem.SolidBodyGravity(field, [0, 0, 0], 0)

    region2 = fem.RegionQuadBoundary(mesh, mask=points, ensure_3d=True)
    field2 = fem.FieldContainer([fem.FieldAxisymmetric(region2, dim=2)])
    pressure = fem.SolidBodyPressure(field2, pressure=0.0)

    step = fem.Step(
        items=[body, load, gravity, pressure],
        ramp={
            bounds["move"]: fem.math.linsteps([0, 1], num=10),
            load: np.zeros((11, 2)),
            pressure: np.zeros(11),
            gravity: np.zeros((11, 3)),
        },
        boundaries=bounds,
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
    job.evaluate(filename="result.xdmf", x0=field)


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
    job.evaluate()


def test_noramp():
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldsMixed(region, n=1)

    umat = fem.LinearElastic(E=1, nu=0.3)
    solid = fem.SolidBody(umat, field)
    bounds = fem.dof.uniaxial(field)[0]

    step = fem.Step(items=[solid], ramp=None, boundaries=bounds)
    job = fem.Job(steps=[step])
    job.evaluate()


if __name__ == "__main__":
    test_job()
    test_job_xdmf()
    test_job_xdmf_global_field()
    test_curve()
    test_curve2()
    test_curve_custom_items()
    test_empty()
    test_noramp()

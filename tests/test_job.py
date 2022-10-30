# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 00:27:38 2022

@author: z0039mte
"""

import numpy as np
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


def test_job():

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()


def test_job_xdmf():

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate(filename="result.xdmf")


def test_job_xdmf_global_field():

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate()

    field, step = pre()
    job = fem.Job(steps=[step])
    job.evaluate(filename="result.xdmf", x0=field)


def test_curve():

    field, step = pre()

    curve = fem.CharacteristicCurve(steps=[step], boundary=step.boundaries["move"])
    curve.plot(xaxis=0, yaxis=0)

    stretch = 1 + np.array(curve.x)[:, 0]
    area = 1**2 * np.pi
    force = (stretch - 1 / stretch**2) * area

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


if __name__ == "__main__":
    test_job()
    test_job_xdmf()
    test_job_xdmf_global_field()
    test_curve()
    test_curve2()

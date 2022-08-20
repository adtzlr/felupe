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

    step = fem.Step(
        items=[body],
        ramp={bounds["move"]: fem.math.linsteps([0, 1], num=10)},
        bounds=bounds,
    )

    return step


def test_job():

    step = pre()

    job = fem.Job(steps=[step])
    job.evaluate()


def test_curve():

    step = pre()

    curve = fem.CharacteristicCurve(steps=[step], boundary=step.bounds["move"])
    curve.plot(xaxis=0, yaxis=0)

    stretch = 1 + np.array(curve.x)[:, 0]
    area = 1 ** 2 * np.pi
    force = (stretch - 1 / stretch ** 2) * area

    assert np.allclose(np.array(curve.y)[:, 0], force, rtol=0.01)


if __name__ == "__main__":
    test_job()
    test_curve()

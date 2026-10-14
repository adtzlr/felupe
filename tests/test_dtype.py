# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import felupe as fem


def test_dtype(dtype=np.float32, tol=1e-3):
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh).astype(dtype, copy=False)
    displacement = fem.Field(region, dim=3, dtype=dtype)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == dtype
    assert field.extract(grad=False)[0].dtype == dtype

    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

    umat = fem.LinearElastic(E=1, nu=0.3)
    solid = fem.SolidBody(umat, field)
    step = fem.Step([solid], boundaries=boundaries)
    job = fem.Job([step]).evaluate(tol=tol)

    assert np.allclose([norm[-1] for norm in job.fnorms], 0, atol=tol)
    assert displacement.values.dtype == dtype


def test_dtype_axi():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh).astype(np.float32)
    displacement = fem.FieldAxisymmetric(region, dtype=np.float32)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == np.float32
    assert field.extract(grad=False)[0].dtype == np.float32


def test_dtype_planestrain():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh, hess=True).astype(np.float32)
    displacement = fem.FieldPlaneStrain(region, dtype=np.float32)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == np.float32
    assert field.extract(grad=False)[0].dtype == np.float32


if __name__ == "__main__":
    test_dtype()
    test_dtype_axi()
    test_dtype_planestrain()

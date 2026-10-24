# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import felupe as fem


def test_free_vibration():
    mesh = fem.Cube(a=(0, 0, 5), b=(50, 100, 30), n=(3, 6, 4))
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    umat = fem.NeoHooke(mu=1, bulk=2)
    solid = fem.SolidBody(umat=umat, field=field, density=1.5e-9)

    job = fem.FreeVibration([solid]).evaluate()
    new_field, frequency = job.extract(n=-1, inplace=False)


def test_free_vibration_mixed():
    meshes = [
        fem.Cube(a=(0, 0, 30), b=(50, 100, 35), n=(3, 6, 2)),
        fem.Cube(a=(0, 0, 5), b=(50, 100, 30), n=(3, 6, 4)),
        fem.Cube(a=(0, 0, 0), b=(50, 100, 5), n=(3, 6, 2)),
    ]
    container = fem.MeshContainer(meshes, merge=True)
    mesh = container.stack()

    regions = [fem.RegionHexahedron(m) for m in container.meshes]
    fields = [
        fem.FieldsMixed(regions[0], n=1),
        fem.FieldsMixed(regions[1], n=3),
        fem.FieldsMixed(regions[2], n=1),
    ]

    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3), *fields[1][1:]])

    boundaries = dict(left=fem.Boundary(field[0], fx=0))
    rubber = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    steel = fem.LinearElasticLargeStrain(2.1e5, 0.3)
    solids = [
        fem.SolidBody(umat=steel, field=fields[0], density=7.85e-9),
        fem.SolidBody(umat=rubber, field=fields[1], density=1.5e-9),
        fem.SolidBody(umat=steel, field=fields[2], density=7.85e-9),
    ]

    job = fem.FreeVibration(solids, boundaries).evaluate(x0=field)
    new_field, frequency = job.extract(x0=field, n=-1, inplace=False)


def test_free_vibration_axisymmetric():
    "The mass matrix and the free vibration of axisymmetric solid bodies."

    mesh = fem.Rectangle(a=(0, 1), b=(2, 3), n=(5, 5))  # axial x, radius 1 <= R <= 3
    region = fem.RegionQuad(mesh)
    density = 2.5

    # the mass of the revolved body (a tube), density * 2 pi * int R dA
    mass = density * 2 * np.pi * 8.0

    for solid in [
        fem.SolidBody(
            umat=fem.NeoHookeCompressible(mu=1.0, lmbda=2.0),
            field=fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)]),
            density=density,
        ),
        fem.SolidBodyNearlyIncompressible(
            umat=fem.NeoHooke(mu=1.0),
            field=fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)]),
            bulk=5.0,
            density=density,
        ),
    ]:
        M = solid.assemble.mass().toarray()

        assert np.allclose(M, M.T)
        assert np.isclose(M[0::2, 0::2].sum(), mass)  # axial direction
        assert np.isclose(M[1::2, 1::2].sum(), mass)  # radial direction
        assert np.allclose(M[0::2, 1::2], 0)

    # the lowest eigenfrequency of the axisymmetric model, clamped at x = 0
    umat = fem.NeoHookeCompressible(mu=1.0, lmbda=2.0)
    field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
    solid = fem.SolidBody(umat=umat, field=field, density=density)
    boundaries = {"fixed": fem.Boundary(field[0], fx=0)}

    job = fem.FreeVibration([solid], boundaries).evaluate(k=1)
    new_field, frequency = job.extract(n=0, inplace=False)

    # the axisymmetric mode is also an eigenmode of the revolved 3d-model
    mesh3d = mesh.revolve(n=37, phi=360).sweep()
    field3d = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh3d), dim=3)])
    solid3d = fem.SolidBody(umat=umat, field=field3d, density=density)
    boundaries3d = {"fixed": fem.Boundary(field3d[0], fx=0)}

    job3d = fem.FreeVibration([solid3d], boundaries3d).evaluate(k=12)
    frequencies3d = np.sqrt(job3d.eigenvalues) / (2 * np.pi)

    assert np.min(np.abs(frequencies3d / frequency - 1)) < 5e-3


def plate(a=(0, 0), b=(5, 1), n=(11, 3)):
    region = fem.RegionQuad(fem.Rectangle(a=a, b=b, n=n))
    return fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])


def test_free_vibration_sigma():
    field = plate()
    solid = fem.SolidBody(fem.LinearElastic(E=2.5, nu=0.25), field, density=1.0)
    boundaries = {"left": fem.Boundary(field[0], fx=0)}

    job = fem.FreeVibration([solid], boundaries).evaluate(k=3)
    eigenvalues = job.eigenvalues

    # the default shift is sigma=0
    job = fem.FreeVibration([solid], boundaries).evaluate(k=3, sigma=0.0)
    assert np.allclose(job.eigenvalues, eigenvalues)

    # a given shift returns the eigenvalue next to it
    job = fem.FreeVibration([solid], boundaries).evaluate(k=1, sigma=eigenvalues[2])
    assert np.allclose(job.eigenvalues, eigenvalues[2])


def test_free_vibration_merged():
    "The top-level field container of merged field containers is used by default."

    fields = [
        fem.FieldsMixed(fem.RegionQuad(fem.Rectangle(n=4)), n=3, planestrain=True),
        fem.FieldsMixed(
            fem.RegionQuad(fem.Rectangle(a=(1, 0), b=(2, 1), n=4)),
            n=3,
            planestrain=True,
        ),
    ]
    x0 = fem.field.merge(fields)

    umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=2))
    solids = [fem.SolidBody(umat, field, density=1.0) for field in fields]
    boundaries = {"left": fem.Boundary(x0[0], fx=0)}

    job = fem.FreeVibration(solids, boundaries).evaluate(x0=x0, k=3)
    eigenvalues = job.eigenvalues

    # without x0, the top-level field container of the first item is used
    job = fem.FreeVibration(solids, boundaries).evaluate(k=3)
    assert np.allclose(job.eigenvalues, eigenvalues)

    # the eigenvector is stored in the field container of evaluate()
    field, frequency = job.extract(n=0)
    assert field is x0
    assert np.isclose(frequency, np.sqrt(eigenvalues[0]) / (2 * np.pi))

    new_field, frequency = job.extract(n=0, inplace=False)
    assert new_field is not x0
    assert new_field.fieldsizes == x0.fieldsizes
    assert np.allclose(new_field[0].values, x0[0].values)


def test_free_vibration_multiplier():
    "The multiplier of an item scales its stiffness and its mass matrix."

    fields = [plate(b=(2.5, 1), n=(6, 3)), plate(a=(2.5, 0), b=(5, 1), n=(6, 3))]
    x0 = fem.field.merge(fields)
    boundaries = {"left": fem.Boundary(x0[0], fx=0)}

    def solid(field, E=1.0, density=1.0, multiplier=None):
        umat = fem.LinearElastic(E=E, nu=0.3)
        return fem.SolidBody(umat, field, density=density, multiplier=multiplier)

    # the right half with a multiplier of 2
    items = [solid(fields[0]), solid(fields[1], multiplier=2.0)]
    job = fem.FreeVibration(items, boundaries).evaluate(k=4)

    # the right half with twice the stiffness and twice the density
    items = [solid(fields[0]), solid(fields[1], E=2.0, density=2.0)]
    job_ref = fem.FreeVibration(items, boundaries).evaluate(k=4)

    # the right half without a multiplier (different eigenvalues)
    items = [solid(fields[0]), solid(fields[1])]
    job_none = fem.FreeVibration(items, boundaries).evaluate(k=4)

    assert np.allclose(job.eigenvalues, job_ref.eigenvalues)
    assert not np.allclose(job.eigenvalues, job_none.eigenvalues)


if __name__ == "__main__":
    test_free_vibration()
    test_free_vibration_mixed()
    test_free_vibration_axisymmetric()
    test_free_vibration_sigma()
    test_free_vibration_merged()
    test_free_vibration_multiplier()

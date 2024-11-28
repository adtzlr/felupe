import numpy as np
import felupe as fem


def test_free_vibration():
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

    assert np.isclose(new_field[0].values.max(), 358.08157)


if __name__ == "__main__":
    test_free_vibration()

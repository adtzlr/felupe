# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 01:40:56 2022

@author: z0039mte
"""

import numpy as np

import felupe as fem


def test_composite():
    n = 5
    mesh = fem.Cube(n=n)
    region = fem.RegionHexahedron(mesh)

    points = np.arange(mesh.npoints)[
        np.logical_or.reduce(
            (
                mesh.points[:, 0] == 0,
                mesh.points[:, 0] == 0 + 1 / (n - 1),
                mesh.points[:, 0] == 0.5 - 1 / (n - 1) / 2,
                mesh.points[:, 0] == 0.5 + 1 / (n - 1) / 2,
                mesh.points[:, 0] == 1 - 1 / (n - 1),
                mesh.points[:, 0] == 1,
            )
        )
    ]
    cells = np.isin(mesh.cells, points).sum(1) == mesh.cells.shape[1]

    mesh_rubber = mesh.copy()
    mesh_rubber.update(cells=mesh_rubber.cells[~cells])

    mesh_reinforced = mesh.copy()
    mesh_reinforced.update(cells=mesh_reinforced.cells[cells])

    region_rubber = fem.RegionHexahedron(mesh_rubber)
    field_rubber = fem.FieldsMixed(
        region_rubber,
        n=3,
        offset=0,
        npoints=mesh_rubber.ncells + mesh_reinforced.ncells,
    )

    region_reinforced = fem.RegionHexahedron(mesh_reinforced)
    field_reinforced = fem.FieldsMixed(
        region_reinforced,
        n=3,
        offset=mesh_rubber.ncells,
        npoints=mesh_rubber.ncells + mesh_reinforced.ncells,
    )

    field = fem.FieldsMixed(
        region, n=3, npoints=mesh_rubber.ncells + mesh_reinforced.ncells
    )

    boundaries, loadcase = fem.dof.uniaxial(field, move=-0.1)

    nh1 = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    nh2 = fem.ThreeFieldVariation(fem.NeoHooke(mu=5000, bulk=5000))

    rubber = fem.SolidBody(nh1, field_rubber)
    reinforced = fem.SolidBody(nh2, field_reinforced)

    res = fem.newtonrhapson(field, items=[rubber, reinforced], **loadcase)

    fem.save(region, res.x)


def test_composite_planestrain():
    n = 5
    mesh = fem.Rectangle(n=n)
    region = fem.RegionQuad(mesh)

    points = np.arange(mesh.npoints)[
        np.logical_or.reduce(
            (
                mesh.points[:, 0] == 0,
                mesh.points[:, 0] == 0 + 1 / (n - 1),
                mesh.points[:, 0] == 0.5 - 1 / (n - 1) / 2,
                mesh.points[:, 0] == 0.5 + 1 / (n - 1) / 2,
                mesh.points[:, 0] == 1 - 1 / (n - 1),
                mesh.points[:, 0] == 1,
            )
        )
    ]
    cells = np.isin(mesh.cells, points).sum(1) == mesh.cells.shape[1]

    mesh_rubber = mesh.copy()
    mesh_rubber.update(cells=mesh_rubber.cells[~cells])

    mesh_reinforced = mesh.copy()
    mesh_reinforced.update(cells=mesh_reinforced.cells[cells])

    region_rubber = fem.RegionQuad(mesh_rubber)
    field_rubber = fem.FieldsMixed(
        region_rubber,
        n=3,
        offset=0,
        planestrain=True,
        npoints=mesh_rubber.ncells + mesh_reinforced.ncells,
    )

    region_reinforced = fem.RegionQuad(mesh_reinforced)
    field_reinforced = fem.FieldsMixed(
        region_reinforced,
        n=3,
        offset=mesh_rubber.ncells,
        planestrain=True,
        npoints=mesh_rubber.ncells + mesh_reinforced.ncells,
    )

    field = fem.FieldsMixed(
        region,
        n=3,
        planestrain=True,
        npoints=mesh_rubber.ncells + mesh_reinforced.ncells,
    )

    boundaries, loadcase = fem.dof.uniaxial(field, move=-0.1)

    nh1 = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    nh2 = fem.ThreeFieldVariation(fem.NeoHooke(mu=5000, bulk=5000))

    rubber = fem.SolidBody(nh1, field_rubber)
    reinforced = fem.SolidBody(nh2, field_reinforced)

    res = fem.newtonrhapson(field, items=[rubber, reinforced], **loadcase)

    fem.save(region, res.x)


if __name__ == "__main__":
    test_composite()
    test_composite_planestrain()

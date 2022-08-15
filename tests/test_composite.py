# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 01:40:56 2022

@author: z0039mte
"""

import felupe as fe
import numpy as np


def test_composite():

    n = 10
    mesh = fe.Cube(n=n)
    region = fe.RegionHexahedron(mesh)

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
    mesh_rubber.update(mesh_rubber.cells[~cells])

    mesh_reinforced = mesh.copy()
    mesh_reinforced.update(mesh_reinforced.cells[cells])

    region_rubber = fe.RegionHexahedron(mesh_rubber)
    field_rubber = fe.FieldsMixed(
        region_rubber,
        n=3,
        offset=0,
        npoints=mesh_rubber.ncells + mesh_reinforced.ncells,
    )

    region_reinforced = fe.RegionHexahedron(mesh_reinforced)
    field_reinforced = fe.FieldsMixed(
        region_reinforced,
        n=3,
        offset=mesh_rubber.ncells,
        npoints=mesh_rubber.ncells + mesh_reinforced.ncells,
    )

    field = fe.FieldsMixed(
        region, n=3, npoints=mesh_rubber.ncells + mesh_reinforced.ncells
    )

    boundaries, loadcase = fe.dof.uniaxial(field, move=-0.1)

    nh1 = fe.ThreeFieldVariation(fe.NeoHooke(mu=1, bulk=5000))
    nh2 = fe.ThreeFieldVariation(fe.NeoHooke(mu=5000, bulk=5000))

    rubber = fe.SolidBody(nh1, field_rubber)
    reinforced = fe.SolidBody(nh2, field_reinforced)

    res = fe.newtonrhapson(field, items=[rubber, reinforced], **loadcase)

    fe.save(region, res.x)


if __name__ == "__main__":
    test_composite()

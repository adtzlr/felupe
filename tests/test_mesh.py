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
import pytest
import numpy as np
import felupe as fe


def test_meshes():

    m = fe.Mesh(
        points=np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        ),
        cells=np.array([[0, 1, 2]]),
        cell_type="triangle",
    )

    fe.mesh.convert(m, order=0)
    fe.mesh.convert(m, order=0, calc_points=True)
    fe.mesh.convert(m, order=2)
    fe.mesh.convert(m, order=2, calc_midfaces=True)

    m = fe.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        cells=np.array([[0, 1, 2, 3]]),
        cell_type="tetra",
    )

    fe.mesh.convert(m, order=0)
    fe.mesh.convert(m, order=0, calc_points=True)
    fe.mesh.convert(m, order=2)
    fe.mesh.convert(m, order=2, calc_midfaces=True)

    m = fe.mesh.Line(n=5)
    assert m.points.shape == (5, 1)
    assert m.cells.shape == (4, 2)

    mr = fe.mesh.revolve(m, n=11, phi=180, axis=2)
    assert mr.ncells == 4 * 10

    m = fe.Rectangle(a=(-1.2, -2), b=(2, 3.1), n=(4, 9))
    assert m.points.shape == (4 * 9, 2)
    assert m.cells.shape == (3 * 8, 4)

    fe.mesh.convert(m, order=0)
    fe.mesh.convert(m, order=0, calc_points=True)
    fe.mesh.convert(m, order=2)
    fe.mesh.convert(m, order=2, calc_midfaces=True)

    mm = m.copy()
    mm.cell_type = "fancy"

    with pytest.raises(NotImplementedError):
        fe.mesh.convert(mm, order=2)

    with pytest.raises(NotImplementedError):
        fe.mesh.convert(m, order=1)

    mr1 = fe.mesh.revolve(m, n=11, phi=180, axis=0)
    fe.mesh.revolve(m.points, m.cells, m.cell_type, n=11, phi=180, axis=0)
    fe.mesh.revolve(m.points, m.cells, m.cell_type, n=11, phi=360, axis=0)

    mr2 = fe.mesh.revolve(m, phi=np.linspace(0, 180, 11), axis=0)

    assert np.allclose(mr1.points, mr2.points)

    with pytest.raises(ValueError):
        fe.mesh.revolve(m.points, m.cells, m.cell_type, n=11, phi=361, axis=0)

    fe.mesh.expand(m.points, m.cells, m.cell_type)
    fe.mesh.expand(m.points, m.cells, cell_type=m.cell_type)
    fe.mesh.expand(m.points, cells=m.cells, cell_type=m.cell_type)
    fe.mesh.expand(points=m.points, cells=m.cells, cell_type=m.cell_type)
    fe.mesh.expand(m)

    me1 = fe.mesh.expand(m, n=3, z=1)
    me2 = fe.mesh.expand(m, n=3, z=1.0)
    me3 = fe.mesh.expand(m, z=np.linspace(0, 1, 3))

    assert np.allclose(me1.points, me2.points)
    assert np.allclose(me2.points, me3.points)

    m = fe.Cube(a=(-1, -2, -0.5), b=(2, 3.1, 1), n=(4, 9, 5))
    assert m.points.shape == (4 * 9 * 5, 3)
    assert m.cells.shape == (3 * 8 * 4, 8)

    fe.mesh.convert(m, order=2, calc_midfaces=True, calc_midvolumes=True)

    with pytest.raises(KeyError):
        fe.mesh.expand(m.points, m.cells, m.cell_type)

    with pytest.raises(KeyError):
        fe.mesh.revolve(m.points, m.cells, m.cell_type)

    fe.mesh.convert(m, order=2, calc_midfaces=True, calc_midvolumes=True)

    fe.mesh.rotate(m, angle_deg=10, axis=0, center=None)
    fe.mesh.rotate(m.points, m.cells, m.cell_type, angle_deg=10, axis=0, center=None)
    fe.mesh.rotate(m, angle_deg=10, axis=1, center=[0, 0, 0])

    fe.mesh.CubeArbitraryOrderHexahedron()
    fe.mesh.RectangleArbitraryOrderQuad()

    m = fe.Rectangle(n=5)
    m.points = np.vstack((m.points, [10, 10]))
    assert m.points.shape == (26, 2)
    assert m.cells.shape == (16, 4)

    m_dg = m.disconnect()
    assert m_dg.dim == m.dim
    assert m_dg.npoints == m.cells.size

    fe.mesh.sweep(m)
    fe.mesh.sweep(m.points, m.cells, m.cell_type, decimals=4)

    m.as_meshio(point_data={"data": m.points}, cell_data={"cell_data": [m.cells[:, 0]]})
    m.save()

    m.cell_type = None
    with pytest.raises(Exception):
        m.save()


def test_mirror():

    for kwargs in [
        dict(axis=None, normal=[1, 0, 0]),
        dict(axis=None, normal=[1, 1, 0]),
        dict(axis=None, normal=[1, 1, 1]),
        dict(axis=None, normal=[-1, 1, 0]),
        dict(axis=None, normal=[1, -5, -3]),
        dict(axis=0, normal=[]),
        dict(axis=1, normal=[]),
        dict(axis=2, normal=[]),
    ]:

        axis = kwargs["axis"]

        if axis is None or axis < 1:

            m = fe.mesh.Line()
            r = fe.Region(m, fe.Line(), fe.GaussLegendre(1, 1))
            n = fe.mesh.mirror(m, **kwargs)
            s = fe.Region(n, fe.Line(), fe.GaussLegendre(1, 1))
            assert np.isclose(r.dV.sum(), s.dV.sum())

        if axis is None or axis < 2:

            m = fe.Rectangle()
            r = fe.RegionQuad(m)
            n = fe.mesh.mirror(m, **kwargs)
            s = fe.RegionQuad(n)
            assert np.isclose(r.dV.sum(), s.dV.sum())

            m = fe.Mesh(
                points=np.array(
                    [
                        [0, 0],
                        [1, 0],
                        [0, 1],
                    ]
                ),
                cells=np.array([[0, 1, 2]]),
                cell_type="triangle",
            )
            r = fe.RegionTriangle(m)
            n = fe.mesh.mirror(m, **kwargs)
            s = fe.RegionTriangle(n)
            assert np.isclose(r.dV.sum(), s.dV.sum())

        m = fe.Cube()
        r = fe.RegionHexahedron(m)
        n = fe.mesh.mirror(m, **kwargs)
        s = fe.RegionHexahedron(n)
        assert np.isclose(r.dV.sum(), s.dV.sum())

        m = fe.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            cells=np.array([[0, 1, 2, 3]]),
            cell_type="tetra",
        )
        r = fe.RegionTetra(m)
        n = fe.mesh.mirror(m, **kwargs)
        s = fe.RegionTetra(n)
        assert np.isclose(r.dV.sum(), s.dV.sum())


def test_triangulate():

    m = fe.Rectangle(n=3)
    n = fe.mesh.triangulate(m)

    rm = fe.RegionQuad(m)
    rn = fe.RegionTriangle(n)

    assert np.isclose(rm.dV.sum(), rn.dV.sum())

    for mode in [0, 3]:
        m = fe.Cube(n=3)
        n = fe.mesh.triangulate(m, mode=mode)

        rm = fe.RegionHexahedron(m)
        rn = fe.RegionTetra(n)

        assert np.isclose(rm.dV.sum(), rn.dV.sum())

    with pytest.raises(NotImplementedError):
        n = fe.mesh.triangulate(m, mode=-1)


def test_runouts():

    m = fe.Rectangle(n=3)

    n = fe.mesh.runouts(m, values=[0.0], axis=0, centerpoint=[0, 0])
    assert n.points[:, 1].max() == m.points[:, 1].max()

    n = fe.mesh.runouts(m, values=[0.1], axis=0, centerpoint=[0, 0])
    assert n.points[:, 1].max() == m.points[:, 1].max() * 1.1

    mask = np.zeros(m.npoints, dtype=bool)
    n = fe.mesh.runouts(m, values=[0.1], axis=0, centerpoint=[0, 0], mask=mask)
    assert n.points[:, 1].max() == m.points[:, 1].max()

    x = [0.5, 0.5]
    n = fe.mesh.runouts(m, values=[0.1], axis=0, centerpoint=x)
    assert (n.points - x)[:, 1].min() == (m.points - x)[:, 1].min() * 1.1
    assert (n.points - x)[:, 1].max() == (m.points - x)[:, 1].max() * 1.1


def test_concatenate():

    m = fe.Rectangle(n=3)

    n = fe.mesh.concatenate([m, m, m])

    assert n.npoints == 3 * m.npoints
    assert n.ncells == 3 * m.ncells


def test_grid():

    m = fe.Rectangle(b=(10, 3), n=(4, 5))

    x = np.linspace(0, 10, 4)
    y = np.linspace(0, 3, 5)
    n = fe.mesh.Grid(x, y)

    assert np.allclose(m.points, n.points)
    assert np.allclose(m.cells, n.cells)


def test_grid_1d():

    m = fe.mesh.Line(b=10, n=6)

    x = np.linspace(0, 10, 6)
    n = fe.mesh.Grid(x)

    assert np.allclose(m.points, n.points)
    assert np.allclose(m.cells, n.cells)


def test_container():

    mesh_1 = fe.Rectangle()
    mesh_2 = fe.Rectangle(a=(1, 0), b=(2, 1))
    mesh_3 = fe.mesh.triangulate(fe.Rectangle(a=(2, 0), b=(3, 1)))

    for merge in [False, True]:
        container = fe.MeshContainer([mesh_1, mesh_2], merge=merge)

    mesh_1
    container

    print(mesh_1)
    print(container)

    assert container.points.shape[1] == container.dim

    container.append(mesh_3)
    print(container.as_meshio())

    m_1 = container.pop(0)

    assert m_1.ncells == 1
    assert len(container.cells()) == 2

    print(container.copy())

    container += mesh_1
    container[2]

    for combined in [False, True]:
        print(container.as_meshio(combined=combined))


def test_read(filename="tests/mesh.bdf"):

    mesh = fe.mesh.read(filename=filename, dim=2)[0]
    assert mesh.dim == 2

    mesh = fe.mesh.read(filename=filename, dim=None)[0]
    assert mesh.dim == 3

    mesh = fe.mesh.read(filename=filename, cellblock=0)[0]
    assert mesh.dim == 3


if __name__ == "__main__":
    test_meshes()
    test_mirror()
    test_triangulate()
    test_runouts()
    test_concatenate()
    test_grid()
    test_grid_1d()
    test_container()
    test_read(filename="mesh.bdf")

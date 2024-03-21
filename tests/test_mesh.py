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
import numpy as np
import pytest

import felupe as fem


def test_meshes():
    m = fem.Mesh(
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

    n = m.copy()
    n.update(points=m.points, cells=m.cells, cell_type="my-fancy-cell-type")

    assert n.cell_type == "my-fancy-cell-type"

    fem.mesh.convert(m, order=0)
    fem.mesh.convert(m, order=0, calc_points=True)
    fem.mesh.convert(m, order=2)
    fem.mesh.convert(m, order=2, calc_midfaces=True)
    m.convert(order=2)

    m = fem.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        cells=np.array([[0, 1, 2, 3]]),
        cell_type="tetra",
    )

    m.dual()
    fem.mesh.dual(m, disconnect=False)
    fem.mesh.dual(m, calc_points=True)

    rect = fem.Rectangle(n=5).add_midpoints_edges()
    rect_dual = rect.dual(points_per_cell=1, disconnect=False)
    assert rect_dual.npoints == 19

    fem.mesh.convert(m, order=0)
    fem.mesh.convert(m, order=0, calc_points=True)
    fem.mesh.convert(m, order=2)
    fem.mesh.convert(m, order=2, calc_midfaces=True)

    m = fem.mesh.Line(n=5)
    assert m.points.shape == (5, 1)
    assert m.cells.shape == (4, 2)

    mr = fem.mesh.revolve(m, n=11, phi=180, axis=2)
    mr = m.revolve(n=11, phi=180, axis=2)
    assert mr.ncells == 4 * 10

    m = fem.Rectangle(a=(-1.2, -2), b=(2, 3.1), n=(4, 9))
    assert m.points.shape == (4 * 9, 2)
    assert m.cells.shape == (3 * 8, 4)

    fem.mesh.convert(m, order=0)
    fem.mesh.convert(m, order=0, calc_points=True)
    fem.mesh.convert(m, order=2)
    fem.mesh.convert(m, order=2, calc_midfaces=True)

    mm = m.copy()
    mm.cell_type = "fancy"

    with pytest.raises(NotImplementedError):
        fem.mesh.convert(mm, order=2)

    with pytest.raises(NotImplementedError):
        fem.mesh.convert(m, order=1)

    mr1 = fem.mesh.revolve(m, n=11, phi=180, axis=0)
    fem.mesh.revolve(m.points, m.cells, m.cell_type, n=11, phi=180, axis=0)
    fem.mesh.revolve(m.points, m.cells, m.cell_type, n=11, phi=360, axis=0)

    mr2 = fem.mesh.revolve(m, phi=np.linspace(0, 180, 11), axis=0)

    assert np.allclose(mr1.points, mr2.points)

    fem.mesh.revolve(m.points, m.cells, m.cell_type, n=11, phi=361, axis=0)

    fem.mesh.expand(m.points, m.cells, m.cell_type)
    fem.mesh.expand(m.points, m.cells, cell_type=m.cell_type)
    fem.mesh.expand(m.points, cells=m.cells, cell_type=m.cell_type)
    fem.mesh.expand(points=m.points, cells=m.cells, cell_type=m.cell_type)
    fem.mesh.expand(m)
    m.expand()

    me1 = fem.mesh.expand(m, n=3, z=1)
    me2 = fem.mesh.expand(m, n=3, z=1.0)
    me3 = fem.mesh.expand(m, z=np.linspace(0, 1, 3))

    assert np.allclose(me1.points, me2.points)
    assert np.allclose(me2.points, me3.points)

    m = fem.Cube(a=(-1, -2, -0.5), b=(2, 3.1, 1), n=(4, 9, 5))
    assert m.points.shape == (4 * 9 * 5, 3)
    assert m.cells.shape == (3 * 8 * 4, 8)

    fem.mesh.convert(m, order=2, calc_midfaces=True, calc_midvolumes=True)

    with pytest.raises(KeyError):
        fem.mesh.expand(m.points, m.cells, m.cell_type)

    with pytest.raises(KeyError):
        fem.mesh.revolve(m.points, m.cells, m.cell_type)

    m.flip()
    fem.mesh.flip(m, mask=[0, 1])

    fem.mesh.convert(m, order=2, calc_midfaces=True, calc_midvolumes=True)

    fem.mesh.rotate(m, angle_deg=10, axis=0, center=None)
    fem.mesh.rotate(m.points, m.cells, m.cell_type, angle_deg=10, axis=0, center=None)
    fem.mesh.rotate(m, angle_deg=10, axis=1, center=[0, 0, 0])
    m.rotate(angle_deg=10, axis=0, center=None)

    fem.mesh.translate(m, move=1, axis=1)
    fem.mesh.translate(m.points, m.cells, m.cell_type, move=1, axis=1)
    fem.mesh.translate(m, move=1, axis=1)
    m.translate(move=1, axis=1)

    fem.mesh.CubeArbitraryOrderHexahedron()
    fem.mesh.RectangleArbitraryOrderQuad()

    m = fem.Rectangle(n=5)
    m.points = np.vstack((m.points, [10, 10]))
    assert m.points.shape == (26, 2)
    assert m.cells.shape == (16, 4)

    m_dg = m.disconnect()
    assert m_dg.dim == m.dim
    assert m_dg.npoints == m.cells.size

    m_dg = m.disconnect(points_per_cell=2, calc_points=False)
    assert np.allclose(m_dg.points, 0)
    assert m_dg.npoints == m.ncells * 2
    assert m_dg.cell_type is None

    fem.mesh.sweep(m)
    fem.mesh.sweep(m.points, m.cells, m.cell_type, decimals=4)
    m.sweep()

    fem.mesh.merge_duplicate_points(m)
    fem.mesh.merge_duplicate_points(m.points, m.cells, m.cell_type, decimals=4)
    m.merge_duplicate_points()

    fem.mesh.merge_duplicate_cells(m)
    m.merge_duplicate_cells()

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
            m = fem.mesh.Line()
            r = fem.Region(m, fem.Line(), fem.GaussLegendre(1, 1))
            n = fem.mesh.mirror(m, **kwargs)
            n = m.mirror(**kwargs)
            s = fem.Region(n, fem.Line(), fem.GaussLegendre(1, 1))
            assert np.isclose(r.dV.sum(), s.dV.sum())

        if axis is None or axis < 2:
            m = fem.Rectangle()
            r = fem.RegionQuad(m)
            n = fem.mesh.mirror(m, **kwargs)
            s = fem.RegionQuad(n)
            assert np.isclose(r.dV.sum(), s.dV.sum())

            m = fem.Mesh(
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
            r = fem.RegionTriangle(m)
            n = fem.mesh.mirror(m, **kwargs)
            s = fem.RegionTriangle(n)
            assert np.isclose(r.dV.sum(), s.dV.sum())

        m = fem.Cube()
        m.x, m.y, m.z
        r = fem.RegionHexahedron(m)
        n = fem.mesh.mirror(m, **kwargs)
        s = fem.RegionHexahedron(n)
        assert np.isclose(r.dV.sum(), s.dV.sum())

        m = fem.Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            cells=np.array([[0, 1, 2, 3]]),
            cell_type="tetra",
        )
        r = fem.RegionTetra(m)
        n = fem.mesh.mirror(m, **kwargs)
        s = fem.RegionTetra(n)
        assert np.isclose(r.dV.sum(), s.dV.sum())


def test_triangulate():
    m = fem.Rectangle(n=3)
    n = fem.mesh.triangulate(m)
    n = m.triangulate()

    rm = fem.RegionQuad(m)
    rn = fem.RegionTriangle(n)

    assert np.isclose(rm.dV.sum(), rn.dV.sum())

    for mode in [0, 3]:
        m = fem.Cube(n=3)
        n = fem.mesh.triangulate(m, mode=mode)

        rm = fem.RegionHexahedron(m)
        rn = fem.RegionTetra(n)

        assert np.isclose(rm.dV.sum(), rn.dV.sum())

    with pytest.raises(NotImplementedError):
        n = fem.mesh.triangulate(m, mode=-1)


def test_runouts():
    m = fem.Rectangle(n=3)

    n = m.add_runouts(values=[0.0], axis=0, centerpoint=[0, 0])
    n = fem.mesh.runouts(m, values=[0.0], axis=0, centerpoint=[0, 0])
    assert n.points[:, 1].max() == m.points[:, 1].max()

    n = fem.mesh.runouts(m, values=[0.1], axis=0, centerpoint=[0, 0])
    assert n.points[:, 1].max() == m.points[:, 1].max() * 1.1

    mask = np.zeros(m.npoints, dtype=bool)
    n = fem.mesh.runouts(m, values=[0.1], axis=0, centerpoint=[0, 0], mask=mask)
    assert n.points[:, 1].max() == m.points[:, 1].max()

    x = [0.5, 0.5]
    n = fem.mesh.runouts(m, values=[0.1], axis=0, centerpoint=x)
    assert (n.points - x)[:, 1].min() == (m.points - x)[:, 1].min() * 1.1
    assert (n.points - x)[:, 1].max() == (m.points - x)[:, 1].max() * 1.1

    n = fem.mesh.runouts(m, values=[0.1], axis=0, centerpoint=x, normalize=True)
    assert (n.points - x)[:, 1].min() == (m.points - x)[:, 1].min()
    assert (n.points - x)[:, 1].max() == (m.points - x)[:, 1].max()


def test_concatenate_stack():
    m = fem.Rectangle(n=3)

    n = fem.mesh.concatenate([m, m, m])

    assert n.npoints == 3 * m.npoints
    assert n.ncells == 3 * m.ncells

    p = fem.mesh.stack([m, m, m])

    assert m.npoints == p.npoints
    assert n.ncells == p.ncells


def test_grid():
    m = fem.Rectangle(b=(10, 3), n=(4, 5))

    x = np.linspace(0, 10, 4)
    y = np.linspace(0, 3, 5)
    n = fem.mesh.Grid(x, y)

    assert np.allclose(m.points, n.points)
    assert np.allclose(m.cells, n.cells)


def test_grid_1d():
    m = fem.mesh.Line(b=10, n=6)

    x = np.linspace(0, 10, 6)
    n = fem.mesh.Grid(x)

    assert np.allclose(m.points, n.points)
    assert np.allclose(m.cells, n.cells)


def test_container():
    mesh_1 = fem.Rectangle()
    mesh_2 = fem.Rectangle(a=(1, 0), b=(2, 1))
    mesh_3 = fem.mesh.triangulate(fem.Rectangle(a=(2, 0), b=(3, 1)))

    for merge in [False, True]:
        container = fem.MeshContainer([mesh_1, mesh_2], merge=merge)

    mesh_1
    container

    print(mesh_1)
    print(container)

    assert container.points.shape[1] == container.dim

    container.append(mesh_3)
    print(container.as_meshio())

    with pytest.raises(TypeError):
        container.stack()

    m_1 = container.pop(0)

    assert m_1.ncells == 1
    assert len(container.cells()) == 2

    print(container.copy())

    container += mesh_1
    container[2]

    for combined in [False, True]:
        print(container.as_meshio(combined=combined))


def test_read(filename="tests/mesh.bdf"):
    mesh = fem.mesh.read(filename=filename, dim=2)[0]
    assert mesh.dim == 2

    mesh = fem.mesh.read(filename=filename, dim=None)[0]
    assert mesh.dim == 3

    mesh = fem.mesh.read(filename=filename, cellblock=0)[0]
    assert mesh.dim == 3


def test_read_nocells(filename="tests/mesh_no-cells.bdf"):
    mesh = fem.mesh.read(filename=filename, dim=2)
    assert mesh[0].dim == 2
    assert mesh[0].ncells == 0
    assert mesh[0].cells.shape == (0, 0)

    mesh = fem.mesh.read(filename=filename, dim=None)
    assert mesh[0].dim == 3
    assert mesh[0].ncells == 0
    assert mesh[0].cells.shape == (0, 0)


def test_mesh_methods():
    mesh = fem.Cube()

    m = mesh.collect_edges()
    assert isinstance(m, fem.Mesh)

    m = mesh.collect_faces()
    assert isinstance(m, fem.Mesh)

    m = mesh.collect_volumes()
    assert isinstance(m, fem.Mesh)

    m = mesh.add_midpoints_edges()
    assert isinstance(m, fem.Mesh)

    m = mesh.add_midpoints_faces()
    assert isinstance(m, fem.Mesh)

    m = mesh.add_midpoints_volumes()
    assert isinstance(m, fem.Mesh)


def test_mesh_fill_between():
    # 2d
    phi = np.linspace(1, 0.5, 11) * np.pi / 2

    line = fem.mesh.Line(n=11)
    bottom = line.copy(points=0.5 * np.vstack([np.cos(phi), np.sin(phi)]).T)
    top = line.copy(points=np.vstack([np.linspace(0, 1, 11), np.linspace(1, 1, 11)]).T)

    face1 = bottom.fill_between(top, n=5)
    face2 = fem.mesh.fill_between(bottom, top, n=5)

    assert np.allclose(face1.points, face2.points)

    region = fem.RegionQuad(face1)

    assert np.all(region.dV > 0)

    # 3d
    face = fem.Rectangle(n=(6, 7))

    bottom = face.copy()
    top = face.copy()

    bottom.points = np.hstack([face.points, np.zeros((face.npoints, 1))])
    top.points = np.hstack([face.points, np.ones((face.npoints, 1))])
    top.points[:, 2] += np.random.rand(top.npoints) / 10

    solid1 = bottom.fill_between(top, n=5)
    solid2 = fem.mesh.fill_between(bottom, top, n=5)

    assert np.allclose(solid1.points, solid2.points)

    region = fem.RegionHexahedron(solid1)

    assert np.all(region.dV > 0)


def test_circle():
    centerpoint = [0, 0]
    radius = 1.5
    x0, y0 = centerpoint

    mesh = fem.Circle(
        radius=radius,
        centerpoint=centerpoint,
        n=6,
        sections=[0, 90, 180],
        exponent=2,
        value=0.15,
        decimals=12,
    )

    x, y = mesh.points.T

    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    assert np.all(r - 1e-10 <= 1.5)

    region = fem.RegionQuad(mesh)

    assert np.all(region.dV > 0)

    boundary = fem.RegionQuadBoundary(mesh)

    assert boundary.mesh.ncells == 50


def test_triangle():
    mesh = fem.mesh.Triangle(
        a=(0, 0),
        b=(1, 0),
        c=(0, 1),
        n=5,
    )

    x, y = mesh.points.T

    region = fem.RegionQuad(mesh)

    assert np.all(region.dV > 0)

    boundary = fem.RegionQuadBoundary(mesh)

    assert boundary.mesh.ncells == 24


def test_view():
    mesh = fem.Rectangle(n=6)
    view = mesh.view()
    plotter = mesh.plot(off_screen=True)
    # img = mesh.screenshot(transparent_background=True)
    # ax = mesh.imshow()

    mesh2 = mesh.translate(move=0.8, axis=0)
    container = fem.MeshContainer([mesh, mesh2])
    assert container[0] is container[[0]][0]
    assert container[0] is container[None][0]

    mesh_stacked = container.stack([0])
    mesh_stacked = container.stack()
    plotter = container.plot(off_screen=True)
    # img = container.screenshot(transparent_background=True)
    # ax = container.imshow()


def test_mesh_update():
    mesh = fem.Cube(n=11)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldsMixed(region, n=1)

    boundaries, loadcase = fem.dof.uniaxial(field, axis=0, sym=False, clamped=True)

    new_points = mesh.rotate(angle_deg=-90, axis=2).points

    # either update the mesh and the region via callback
    mesh.update(points=new_points, callback=region.reload)

    # or update the region separately
    mesh.update(points=new_points)
    region.reload(mesh)


def test_modify_corners():
    fem.Rectangle(n=6).modify_corners()
    fem.Cube(n=6).modify_corners()

    with pytest.raises(TypeError):
        fem.Rectangle(n=6).triangulate().modify_corners()

    with pytest.raises(TypeError):
        fem.Cube(n=6).triangulate().modify_corners()


def test_expand():
    point = fem.Point(a=0.0)
    line = point.expand(n=6, z=1)
    rect = line.expand(n=11, z=2)
    cube = rect.expand(n=16, z=3)

    assert np.allclose(cube.points.max(axis=0), [1, 2, 3])


if __name__ == "__main__":
    test_meshes()
    test_mirror()
    test_triangulate()
    test_runouts()
    test_concatenate_stack()
    test_grid()
    test_grid_1d()
    test_container()
    test_read(filename="mesh.bdf")
    test_mesh_methods()
    test_read_nocells(filename="mesh_no-cells.bdf")
    test_mesh_fill_between()
    test_circle()
    test_triangle()
    test_view()
    test_mesh_update()
    test_modify_corners()
    test_expand()

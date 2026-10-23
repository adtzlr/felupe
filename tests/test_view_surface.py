# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import felupe as fem
from felupe.view._surface import FACES, _single_faces, triangulate_surface


def test_single_faces():
    "The outer faces by two integer keys are equal to the faces by numpy.unique."

    def reference(faces):
        keys = np.sort(faces, axis=1)
        _, index, counts = np.unique(
            keys, axis=0, return_index=True, return_counts=True
        )
        return faces[np.sort(index[counts == 1])]

    # wedges by two layers of triangles
    tri = fem.Rectangle(n=4).triangulate()
    layers = [np.hstack([tri.points, np.full((tri.npoints, 1), z)]) for z in [0, 1]]
    cells = np.hstack([tri.cells, tri.cells + tri.npoints])
    wedge = fem.Mesh(np.vstack(layers), cells, "wedge")

    meshes = [fem.Cube(n=5), fem.Cube(n=5).triangulate(), wedge]
    for mesh in meshes:
        for size in [3, 4]:
            local = [f for f in FACES[mesh.cell_type] if len(f) == size]
            if local:
                faces = mesh.cells[:, local].reshape(-1, size)
                assert np.array_equal(faces[_single_faces(faces)], reference(faces))

    assert _single_faces(np.zeros((0, 4), dtype=int)).shape == (0,)


def area(points, triangles):
    "The area of a triangulated surface."

    points = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    a = points[triangles[:, 1]] - points[triangles[:, 0]]
    b = points[triangles[:, 2]] - points[triangles[:, 0]]
    return np.linalg.norm(np.cross(a, b), axis=1).sum() / 2


def length(points, edges):
    "The total length of line segments."

    return np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1).sum()


def wedges():
    "A mesh with wedges by two layers of triangles."

    tri = fem.Rectangle(n=3).triangulate()
    layers = [np.hstack([tri.points, np.full((tri.npoints, 1), z)]) for z in [0, 1]]
    cells = np.hstack([tri.cells, tri.cells + tri.npoints])
    return fem.Mesh(np.vstack(layers), cells, "wedge")


def meshes():
    "Meshes of all supported cell types (with the polynomial order of the cells)."

    cube = fem.Cube(n=3)
    rectangle = fem.Rectangle(n=3)
    convert = fem.mesh.convert

    return {
        "hexahedron": (cube, 1),
        "hexahedron20": (convert(cube, order=2), 2),
        "hexahedron27": (
            convert(cube, order=2, calc_midfaces=True, calc_midvolumes=True),
            2,
        ),
        "tetra": (cube.triangulate(), 1),
        "tetra10": (convert(cube.triangulate(), order=2), 2),
        "quad": (rectangle, 1),
        "quad8": (convert(rectangle, order=2), 2),
        "quad9": (convert(rectangle, order=2, calc_midfaces=True), 2),
        "triangle": (rectangle.triangulate(), 1),
        "triangle6": (convert(rectangle.triangulate(), order=2), 2),
        "VTK_LAGRANGE_HEXAHEDRON": (fem.mesh.CubeArbitraryOrderHexahedron(order=3), 3),
        "VTK_LAGRANGE_QUADRILATERAL": (
            fem.mesh.RectangleArbitraryOrderQuad(order=4),
            4,
        ),
        "line": (fem.mesh.Line(n=4), 1),
        "wedge": (wedges(), 1),
    }


def curved(mesh):
    "A copy of a mesh with smoothly distorted (curved) points."

    points = np.pad(mesh.points, ((0, 0), (0, 3 - mesh.dim)))
    x, y, z = points.T
    shift = 0.1 * np.stack([np.sin(3 * y + z), np.cos(2 * x - z), np.sin(x * y)], 1)
    return mesh.copy(points=(points + shift)[:, : mesh.dim])


def test_triangulate_surface_like_pyvista():
    "The area and the edges of the surface are equal to the surface of PyVista."

    pv = pytest.importorskip("pyvista")

    for name, (mesh, order) in meshes().items():
        surface = triangulate_surface(mesh)
        points = surface.interpolate(mesh.points)

        if name == "wedge":
            grid = pv.UnstructuredGrid(
                {pv.CellType.WEDGE: mesh.cells}, np.asarray(mesh.points, dtype=float)
            )
        else:
            grid = mesh.as_unstructured_grid()

        reference = grid.extract_surface(algorithm=None).triangulate()

        # the unique edges of the linear surface (by the corner points of the faces)
        edges = grid.extract_surface(algorithm=None, nonlinear_subdivision=0)
        edges = edges.extract_all_edges().compute_cell_sizes(length=True)

        if name != "line":
            assert np.isclose(area(points, surface.triangles), reference.area), name

        own = np.pad(points, ((0, 0), (0, 3 - mesh.dim)))
        assert np.isclose(length(own, surface.edges), edges["Length"].sum()), name

        # the points of the surface of PyVista are points of the surface
        for point in reference.points:
            assert np.min(np.linalg.norm(own - point, axis=1)) < 1e-6, name


def test_triangulate_surface_interpolation():
    for name, (mesh, order) in meshes().items():
        mesh = curved(mesh)

        for subdivisions in [None, 1, 3]:
            surface = triangulate_surface(mesh, subdivisions=subdivisions)
            matrix = surface.interpolation
            points = surface.interpolate(mesh.points)

            # a linear field is interpolated exactly (with a partition of unity)
            values = 2.0 + mesh.points @ np.arange(1, mesh.dim + 1)
            assert np.allclose(
                surface.interpolate(values), 2.0 + points @ np.arange(1, mesh.dim + 1)
            )
            assert np.allclose(matrix.sum(axis=1), 1)

            # each point of the surface exists only once
            assert len(np.unique(np.round(points, 10), axis=0)) == len(points), name

            # the edges are unique and all points are used
            edges = np.sort(surface.edges, axis=1)
            assert len(np.unique(edges, axis=0)) == len(edges)
            used = np.unique(np.concatenate([surface.triangles.ravel(), edges.ravel()]))
            assert np.array_equal(used, np.arange(matrix.shape[0])), name

        # with the polynomial order of the cells, the points are the points of the mesh
        surface = triangulate_surface(mesh)
        lagrange = name not in ["hexahedron20", "quad8"]
        selection = surface.interpolation.nnz == surface.interpolation.shape[0]
        assert selection == lagrange, name
        if lagrange:
            assert np.all(surface.interpolation.data == 1.0)


def test_triangulate_surface_counts():
    cube = fem.Cube(n=3)
    surface = triangulate_surface(cube)
    assert surface.interpolation.shape == (26, 27)
    assert surface.triangles.shape == (48, 3)
    assert surface.edges.shape == (48, 2)

    # the triangles are oriented outwards
    points = surface.interpolate(cube.points)
    a, b, c = points[surface.triangles].transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    centers = (a + b + c) / 3
    assert np.all(np.sum(normals * (centers - 0.5), axis=1) > 0)

    # the owners are the cells of the triangles
    region = fem.RegionHexahedron(cube)
    for triangle, owner in zip(points[surface.triangles], surface.owners):
        lower, upper = (
            cube.points[cube.cells[owner]].min(0),
            cube.points[cube.cells[owner]].max(0),
        )
        assert np.all((triangle >= lower - 1e-12) & (triangle <= upper + 1e-12))

    # subdivided quadratic hexahedrons, see the docstring
    surface = triangulate_surface(fem.mesh.convert(cube, order=2), subdivisions=4)
    assert surface.interpolation.shape == (26 + 48 * 3 + 24 * 9, 81)
    assert surface.triangles.shape == (24 * 32, 3)
    assert surface.edges.shape == (48 * 4, 2)

    # all cells of two-dimensional meshes, the edges of lines
    surface = triangulate_surface(fem.Rectangle(n=3))
    assert surface.triangles.shape == (8, 3) and surface.edges.shape == (12, 2)
    surface = triangulate_surface(fem.mesh.Line(n=4), subdivisions=2)
    assert surface.triangles.shape == (0, 3) and surface.edges.shape == (6, 2)

    # without supported cells
    vertex = fem.Mesh(np.zeros((1, 3)), np.zeros((1, 1), dtype=int), "vertex")
    surface = triangulate_surface(vertex)
    assert surface.interpolation.shape == (0, 1)
    assert surface.triangles.shape == (0, 3) and surface.edges.shape == (0, 2)
    assert triangulate_surface([]).interpolation.shape == (0, 0)

    # a mesh without cells
    empty = fem.Mesh(np.zeros((3, 2)), np.zeros((0, 2), dtype=int), "line")
    assert triangulate_surface(empty).interpolation.shape == (0, 3)

    with pytest.raises(ValueError):
        triangulate_surface(cube, subdivisions=0)


def test_triangulate_surface_of_meshes():
    cubes = [fem.Cube(n=3), fem.Cube(n=3).translate(1, axis=0)]

    # meshes with shared points: the faces at the interface are inner faces
    container = fem.MeshContainer(cubes, merge=True)
    surface = triangulate_surface(container.meshes)
    points = surface.interpolate(container.points)
    assert np.isclose(area(points, surface.triangles), 10.0)
    assert surface.interpolation.shape[1] == container.points.shape[0]

    # meshes with own points: the points are stacked
    surface = triangulate_surface(cubes)
    points = surface.interpolate(np.vstack([cube.points for cube in cubes]))
    assert np.isclose(area(points, surface.triangles), 12.0)
    assert surface.interpolation.shape[1] == 2 * 27

    # the owners are numbered over all meshes
    assert surface.owners.max() == 2 * 8 - 1

    # meshes with equal (but not the same) points: the points are shared
    copies = [container.meshes[0], container.meshes[1].copy()]
    assert copies[1].points is not copies[0].points
    surface = triangulate_surface(copies)
    assert surface.interpolation.shape[1] == container.points.shape[0]

    # different cell types with shared points, a quad and two triangles at the
    # interface are not identical (both are kept and hidden by the outer faces)
    hexahedron = fem.Cube(n=2)
    tetra = fem.Cube(n=2).translate(1, axis=0).triangulate()
    container = fem.MeshContainer([hexahedron, tetra], merge=True)
    surface = triangulate_surface(container.meshes)
    assert np.isclose(area(container.points, surface.triangles), 10.0 + 2.0)


def test_triangulate_surface_line_edges():
    "The edges of the line cells are marked (also if they are edges of faces)."

    rectangle = fem.Rectangle(n=3)
    lines = fem.Mesh(rectangle.points, np.array([[0, 4], [4, 8], [0, 1]]), "line")

    surface = triangulate_surface([rectangle, lines])
    assert surface.line_edges.shape == (len(surface.edges),)
    assert np.array_equal(
        np.sort(np.sort(surface.edges[surface.line_edges], axis=1), axis=0),
        [[0, 1], [0, 4], [4, 8]],
    )

    # subdivided edges of quadratic line cells (the corner points first)
    points = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
    line = fem.Mesh(points, np.array([[0, 2, 1], [2, 4, 3]]), "line3")
    surface = triangulate_surface(line, subdivisions=4)
    assert surface.edges.shape == (2 * 4, 2) and surface.line_edges.all()

    # faces without line cells
    assert not triangulate_surface(fem.Cube(n=3)).line_edges.any()


def test_triangulate_surface_curved():
    "The area of a curved surface converges with the number of subdivisions."

    # a quarter of an annulus with points on circles by biquadratic quads
    mesh = fem.mesh.convert(
        fem.Rectangle(a=(1, 0), b=(2, 1), n=3), order=2, calc_midfaces=True
    )
    r, phi = mesh.points[:, 0], mesh.points[:, 1] * np.pi / 2
    mesh = mesh.copy(points=np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1))

    # the area of the isoparametric geometry (the quadrature is exact enough)
    quadrature = fem.GaussLegendre(order=6, dim=2)
    exact = fem.RegionBiQuadraticQuad(mesh, quadrature=quadrature).dV.sum()

    errors = []
    for subdivisions in [2, 4, 8, 16]:
        surface = triangulate_surface(mesh, subdivisions=subdivisions)
        points = surface.interpolate(mesh.points)
        errors.append(abs(area(points, surface.triangles) - exact) / exact)

    # the error of the triangulation is of second order
    assert np.allclose(np.array(errors[:-1]) / errors[1:], 4, rtol=0.05)
    assert errors[-1] < 5e-4

    # the edges on the circles are curved
    surface = triangulate_surface(mesh, subdivisions=8)
    points = surface.interpolate(mesh.points)
    radius = np.linalg.norm(points[np.unique(surface.edges)], axis=1)
    outer = radius > 1.99
    assert np.sum(outer) == 2 * 8 + 1  # two cells with eight segments per edge
    assert np.allclose(radius[outer], 2, rtol=1e-3)

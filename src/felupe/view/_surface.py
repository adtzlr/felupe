# SPDX-License-Identifier: GPL-3.0-or-later

"""The outer surface of meshes for the plotting backends and the dashboard,
independent of PyVista.

:func:`boundary_faces` returns the corner points of the outer faces, e.g. for the
viewport of the dashboard. :func:`triangulate_surface` returns a triangulated surface
with its edges, optionally subdivided, and a sparse matrix which interpolates point
values of the meshes (e.g. the coordinates or the displacements) to the points of the
surface by the shape functions of the finite elements. Hence, the faces of quadratic
or Lagrange cells are curved.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

# local faces (VTK ordering) of the corner points of three-dimensional cells
FACES = {
    "hexahedron": [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ],
    "tetra": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    "wedge": [[0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
    "pyramid": [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
}

# number of corner points of the cells, higher-order cells are drawn by their corners
CORNERS = {
    "hexahedron": ("hexahedron", 8),
    "hexahedron20": ("hexahedron", 8),
    "hexahedron27": ("hexahedron", 8),
    "VTK_LAGRANGE_HEXAHEDRON": ("hexahedron", 8),
    "tetra": ("tetra", 4),
    "tetra10": ("tetra", 4),
    "wedge": ("wedge", 6),
    "wedge15": ("wedge", 6),
    "pyramid": ("pyramid", 5),
    "quad": ("quad", 4),
    "quad8": ("quad", 4),
    "quad9": ("quad", 4),
    "VTK_LAGRANGE_QUADRILATERAL": ("quad", 4),
    "triangle": ("triangle", 3),
    "triangle6": ("triangle", 3),
    "line": ("line", 2),
    "line3": ("line", 2),
    "VTK_LAGRANGE_LINE": ("line", 2),
}


def _unique_faces(faces):
    """Return the faces which belong to only one cell (with their original ordering).
    See :func:`_single_faces`."""

    return faces[_single_faces(faces)]


def _single_faces(faces):
    """Return a mask of the faces which belong to only one cell.

    The sorted point indices of a face are combined into two integer keys (each of at
    most two indices, i.e. for less than about three billion points). A lexical sort of
    the two keys is much faster than a sort of the rows of the faces, which is used by
    ``numpy.unique(faces, axis=0)``.
    """
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)

    keys = np.sort(faces, axis=1).astype(np.int64)
    npoints = int(keys.max()) + 1
    half = (keys.shape[1] + 1) // 2

    key1 = keys[:, 0]
    for j in range(1, half):
        key1 = key1 * npoints + keys[:, j]

    key2 = keys[:, half]
    for j in range(half + 1, keys.shape[1]):
        key2 = key2 * npoints + keys[:, j]

    order = np.lexsort((key2, key1))
    key1, key2 = key1[order], key2[order]

    # the first face of each group of equal faces and the size of the groups
    new = np.ones(len(order), dtype=bool)
    new[1:] = (key1[1:] != key1[:-1]) | (key2[1:] != key2[:-1])
    group = np.cumsum(new) - 1
    single = np.bincount(group)[group] == 1

    mask = np.zeros(len(faces), dtype=bool)
    mask[order[single]] = True

    return mask


def boundary_faces(meshes):
    """Return the triangles, quads and lines of the outer surface of one or more
    meshes with a common points-array, given by the corner points of the cells.

    Parameters
    ----------
    meshes : felupe.Mesh or list of felupe.Mesh
        The mesh or a list of meshes with a common points-array, e.g. the meshes of a
        :class:`~felupe.MeshContainer`. Each mesh has its own cells and cell type.

    Returns
    -------
    dict
        A dict with the arrays ``"tri"`` (n, 3), ``"quad"`` (n, 4) and ``"line"``
        (n, 2) of point indices and the names of the base cell types ``"kind"``
        (``"points"`` if no cell type is supported).

    Notes
    -----
    The faces of the three-dimensional cells of all meshes are collected first. A face
    which belongs to two cells is an inner face, also if the cells belong to different
    meshes, e.g. at the interface of two bodies with merged points. Only identical
    faces are inner faces, i.e. the quad of a hexahedron and the two triangles of
    tetrahedra at a common interface are kept (they are hidden by the outer faces).
    Two-dimensional cells are faces and line cells are lines.
    """

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    faces = {3: [], 4: []}
    cells2d = {3: [], 4: []}
    lines = []
    kinds = []

    for mesh in meshes:
        kind, ncorners = CORNERS.get(mesh.cell_type, (None, None))

        if kind is None:
            continue

        kinds.append(kind)
        corners = np.asarray(mesh.cells)[:, :ncorners]

        if kind in FACES:
            for size in [3, 4]:
                local = [f for f in FACES[kind] if len(f) == size]
                if local:
                    faces[size].append(corners[:, local].reshape(-1, size))
        elif kind == "quad":
            cells2d[4].append(corners)
        elif kind == "triangle":
            cells2d[3].append(corners)
        else:
            lines.append(corners)

    def stack(arrays, size, unique=False):
        if not arrays:
            return np.zeros((0, size), dtype=np.int64)
        stacked = np.concatenate(arrays)
        return _unique_faces(stacked) if unique else stacked

    tri = np.concatenate([stack(faces[3], 3, unique=True), stack(cells2d[3], 3)])
    quad = np.concatenate([stack(faces[4], 4, unique=True), stack(cells2d[4], 4)])
    line = stack(lines, 2)

    kind = ", ".join(dict.fromkeys(kinds)) if kinds else "points"

    return dict(kind=kind, tri=tri, quad=quad, line=line)


def compact(surf, npoints):
    """Return the indices of the points of a surface and the surface with the cells
    renumbered to these points.

    Parameters
    ----------
    surf : dict
        A dict with the arrays ``"tri"``, ``"quad"`` and ``"line"`` of point indices,
        see :func:`boundary_faces`.
    npoints : int
        The number of points of the mesh.

    Returns
    -------
    ndarray
        The sorted indices of the points of the surface. Without cells, e.g. for the
        kind ``"points"``, all points of the mesh are used.
    dict
        The surface with the cells renumbered to the points of the surface.
    """

    cells = [np.asarray(surf[key]).ravel() for key in ["tri", "quad", "line"]]

    if any(c.size for c in cells):
        points = np.unique(np.concatenate(cells))
    else:
        points = np.arange(npoints)

    index = np.zeros(npoints, dtype=np.int64)
    index[points] = np.arange(len(points))
    renumbered = {key: index[np.asarray(surf[key])] for key in ["tri", "quad", "line"]}

    return points, {**surf, **renumbered}


# the ring of the corner points of a cell, which is the face of a two-dimensional cell
RINGS = {"quad": [0, 1, 2, 3], "triangle": [0, 1, 2], "line": [0, 1]}


def _element(cell_type, points_per_cell):
    """Return the finite element of FElupe (or None) and the polynomial order of the
    cells of a cell type, i.e. the number of segments of the edges of the cells."""

    from ..element import (
        ArbitraryOrderLagrange,
        BiQuadraticQuad,
        Hexahedron,
        Line,
        Quad,
        QuadraticHexahedron,
        QuadraticQuad,
        QuadraticTetra,
        QuadraticTriangle,
        Tetra,
        TetraMINI,
        Triangle,
        TriangleMINI,
        TriQuadraticHexahedron,
    )

    elements = {
        ("line", 2): (Line, 1),
        ("triangle", 3): (Triangle, 1),
        ("triangle", 4): (TriangleMINI, 1),
        ("triangle6", 6): (QuadraticTriangle, 2),
        ("tetra", 4): (Tetra, 1),
        ("tetra", 5): (TetraMINI, 1),
        ("tetra10", 10): (QuadraticTetra, 2),
        ("quad", 4): (Quad, 1),
        ("quad8", 8): (QuadraticQuad, 2),
        ("quad9", 9): (BiQuadraticQuad, 2),
        ("hexahedron", 8): (Hexahedron, 1),
        ("hexahedron20", 20): (QuadraticHexahedron, 2),
        ("hexahedron27", 27): (TriQuadraticHexahedron, 2),
    }

    if (cell_type, points_per_cell) in elements:
        element, order = elements[(cell_type, points_per_cell)]
        return element(), order

    # the arbitrary-order Lagrange cells in VTK-ordering (the corner points first)
    dims = {
        "line3": 1,
        "VTK_LAGRANGE_LINE": 1,
        "VTK_LAGRANGE_QUADRILATERAL": 2,
        "VTK_LAGRANGE_HEXAHEDRON": 3,
    }

    if cell_type in dims:
        dim = dims[cell_type]
        order = int(round(points_per_cell ** (1 / dim))) - 1
        if order > 0 and (order + 1) ** dim == points_per_cell:
            return ArbitraryOrderLagrange(order=order, dim=dim), order

    # other cells (e.g. wedges) are interpolated linearly by their corner points
    return None, 1


def polynomial_order(meshes):
    """Return the largest polynomial order of the supported cells of one or more meshes
    (and 1 if there are no supported cells), i.e. the number of segments per edge of
    the faces of :func:`triangulate_surface` without given subdivisions."""

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    orders = [
        _element(str(mesh.cell_type), np.shape(mesh.cells)[1])[1]
        for mesh in meshes
        if str(mesh.cell_type) in CORNERS and len(mesh.cells) > 0
    ]

    return max(orders, default=1)


@dataclass
class _Template:
    """A face, a cell or a line, subdivided into ``n`` segments per edge. The points of
    the subdivision are given by parameters of the ring of the corner points.

    Attributes
    ----------
    params : ndarray
        The parameters of the points in ``[0, 1]`` of shape ``(npoints, 2)``.
    corners : ndarray
        The local corner of the ring of each point (or -1).
    edges : list of tuple
        The edges ``(a, b)`` of the ring by its local corners.
    edge_points : ndarray
        The points on the edges, in the direction from ``a`` to ``b`` (without the
        corner points), of shape ``(nedges, n - 1)``.
    interior : ndarray
        The points in the interior of the face.
    triangles : ndarray
        The triangles of the subdivision, oriented like the ring.
    """

    params: np.ndarray
    corners: np.ndarray
    edges: list
    edge_points: np.ndarray
    interior: np.ndarray
    triangles: np.ndarray

    def weights(self, ring, element, points_per_cell):
        """Return the weights of the points of the cells at the points of the template,
        either by the shape functions of the element or linear in the corner points of
        the ring, of shape ``(npoints, points_per_cell)``."""

        s, t = self.params.T

        if len(ring) == 4:
            corner_weights = [(1 - s) * (1 - t), s * (1 - t), s * t, (1 - s) * t]
        elif len(ring) == 3:
            corner_weights = [1 - s - t, s, t]
        else:
            corner_weights = [1 - s, s]

        corner_weights = np.array(corner_weights).T

        if element is None:
            weights = np.zeros((len(s), points_per_cell))
            weights[:, ring] = corner_weights
        else:
            reference = np.asarray(element.points, dtype=float)
            rst = corner_weights @ reference[ring]
            weights = np.array([element.function(r) for r in rst])

        # clean the round-off errors (the corner points of a cell are exact)
        weights[np.abs(weights) < 1e-12] = 0.0
        exact = np.isclose(weights, 1.0, rtol=0, atol=1e-12)
        weights[exact] = 1.0

        return weights


def _template(corners, n):
    """Return the template of a quad (four corners), a triangle (three corners) or a
    line (two corners) with ``n`` segments per edge."""

    if corners == 4:
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="xy")
        i, j = i.ravel(), j.ravel()
        index = lambda a, b: b * (n + 1) + a
        corner_ij = [(0, 0), (n, 0), (n, n), (0, n)]

        def edge(e, m):
            return [(m, 0), (n, m), (n - m, n), (0, n - m)][e]

        cells = [
            (index(a, b), index(a + 1, b), index(a + 1, b + 1), index(a, b + 1))
            for b in range(n)
            for a in range(n)
        ]
        triangles = [(p, q, r) for p, q, r, s in cells] + [
            (p, r, s) for p, q, r, s in cells
        ]
        ring_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    elif corners == 3:
        pairs = [(a, b) for b in range(n + 1) for a in range(n + 1 - b)]
        i, j = np.array(pairs).T.reshape(2, -1)
        lookup = {pair: k for k, pair in enumerate(pairs)}
        index = lambda a, b: lookup[(a, b)]
        corner_ij = [(0, 0), (n, 0), (0, n)]

        def edge(e, m):
            return [(m, 0), (n - m, m), (0, n - m)][e]

        triangles = [
            (index(a, b), index(a + 1, b), index(a, b + 1))
            for b in range(n)
            for a in range(n - b)
        ] + [
            (index(a + 1, b), index(a + 1, b + 1), index(a, b + 1))
            for b in range(n - 1)
            for a in range(n - 1 - b)
        ]
        ring_edges = [(0, 1), (1, 2), (2, 0)]

    else:
        i, j = np.arange(n + 1), np.zeros(n + 1, dtype=int)
        index = lambda a, b: a
        corner_ij = [(0, 0), (n, 0)]

        def edge(e, m):
            return (m, 0)

        triangles = []
        ring_edges = [(0, 1)]

    npoints = len(i)
    point_corners = np.full(npoints, -1)
    for k, (a, b) in enumerate(corner_ij):
        point_corners[index(a, b)] = k

    edge_points = np.array(
        [[index(*edge(e, m)) for m in range(1, n)] for e in range(len(ring_edges))],
        dtype=np.int64,
    ).reshape(len(ring_edges), n - 1)

    on_ring = np.zeros(npoints, dtype=bool)
    on_ring[point_corners >= 0] = True
    on_ring[edge_points.ravel()] = True

    return _Template(
        params=np.stack([i, j], axis=1) / n,
        corners=point_corners,
        edges=ring_edges,
        edge_points=edge_points,
        interior=np.flatnonzero(~on_ring),
        triangles=np.array(triangles, dtype=np.int64).reshape(-1, 3),
    )


@dataclass
class Surface:
    r"""A triangulated surface of meshes with its edges, given by the points of the
    surface, which are interpolated from the points of the meshes.

    Attributes
    ----------
    interpolation : scipy.sparse.csr_matrix
        The matrix of shape ``(nsurface, npoints)``, which interpolates the values at
        the points of the meshes to the points of the surface by the shape functions of
        the cells, e.g. ``surface.interpolation @ mesh.points``. Without a subdivision
        of linear cells, each row selects a point of the meshes.
    triangles : ndarray
        The point-connectivity of the triangles of shape ``(ntriangles, 3)``, oriented
        outwards for the faces of three-dimensional cells.
    owners : ndarray
        The cells of the triangles of shape ``(ntriangles,)``, numbered over all
        meshes, e.g. to take the values of cell-data.
    faces : ndarray
        The faces of the triangles of shape ``(ntriangles,)``, i.e. the triangles of a
        (subdivided) face of a cell share the same number, e.g. to take a common
        normal vector of a face.
    edges : ndarray
        The point-connectivity of the line segments of the edges of the faces, the
        two-dimensional cells and the line cells of shape ``(nedges, 2)``.
    """

    interpolation: csr_matrix
    triangles: np.ndarray
    owners: np.ndarray
    faces: np.ndarray
    edges: np.ndarray

    def interpolate(self, values):
        """Return the values at the points of the surface, interpolated from given
        values at the points of the meshes of shape ``(npoints, ...)``."""

        values = np.asarray(values)
        shape = values.shape

        return (self.interpolation @ values.reshape(shape[0], -1)).reshape(
            -1, *shape[1:]
        )


def triangulate_surface(meshes, subdivisions=None):
    r"""Return the triangulated outer surface of one or more meshes with its edges.

    Parameters
    ----------
    meshes : felupe.Mesh or list of felupe.Mesh
        The mesh or a list of meshes, e.g. the meshes of a
        :class:`~felupe.MeshContainer`. If all meshes share the same points-array,
        their points are shared, otherwise the points of the meshes are stacked.
    subdivisions : int or None, optional
        The number of segments of each edge of a face (default is None). If None, the
        polynomial order of the cells is used (the largest of all meshes), i.e. the
        faces are triangulated by their points, e.g. by the mid-edge points of
        quadratic cells. More subdivisions show the curved faces of quadratic and
        Lagrange cells by their shape functions.

    Returns
    -------
    Surface
        The triangulated surface with its edges and the matrix which interpolates the
        values at the points of the meshes to the points of the surface.

    Notes
    -----
    The faces of the three-dimensional cells which belong to only one cell are outer
    faces (like for :func:`boundary_faces`). Two-dimensional cells are faces and line
    cells are edges. The points on the edges and the corner points are shared by the
    faces, i.e. each point of the surface exists only once. The values at the points
    of a face are interpolated by the shape functions of the finite element of the
    cell type. Cell types without a finite element of FElupe, e.g. wedges, are
    interpolated linearly by their corner points.

    Examples
    --------
    >>> import felupe as fem
    >>> from felupe.view._surface import triangulate_surface
    >>>
    >>> mesh = fem.mesh.convert(fem.Cube(n=3), order=2)
    >>> surface = triangulate_surface(mesh, subdivisions=4)
    >>> points = surface.interpolate(mesh.points)
    >>> points.shape, surface.triangles.shape, surface.edges.shape
    ((386, 3), (768, 3), (192, 2))
    """

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    meshes = list(meshes)
    shared = all(mesh.points is meshes[0].points for mesh in meshes)

    # the offsets of the points and the cells of the meshes
    npoints = [len(mesh.points) for mesh in meshes]
    point_offsets = np.zeros(len(meshes), dtype=np.int64)
    if not shared:
        point_offsets[1:] = np.cumsum(npoints)[:-1]
    total_points = npoints[0] if shared and meshes else sum(npoints)
    cell_offsets = np.concatenate([[0], np.cumsum([len(m.cells) for m in meshes])])

    # the supported meshes with their cells (by the numbering of all points)
    groups = []
    for mesh, point_offset, cell_offset in zip(meshes, point_offsets, cell_offsets):
        cells = np.asarray(mesh.cells, dtype=np.int64)
        kind, ncorners = CORNERS.get(str(mesh.cell_type), (None, None))

        if kind is not None and len(cells) > 0:
            element, order = _element(str(mesh.cell_type), cells.shape[1])
            groups.append(
                dict(
                    kind=kind,
                    cells=cells + point_offset,
                    cell_offset=cell_offset,
                    element=element,
                    order=order,
                )
            )

    n = subdivisions
    if n is None:
        n = max([group["order"] for group in groups], default=1)
    n = int(n)

    if n < 1:
        raise ValueError("The number of subdivisions must be positive.")

    # the faces, cells and lines with their instances (the cells of the faces)
    faces = []
    for size in [3, 4]:
        candidates = [
            (group, ring)
            for group in groups
            for ring in FACES.get(group["kind"], [])
            if len(ring) == size
        ]

        if candidates:
            keys = np.concatenate(
                [group["cells"][:, ring] for group, ring in candidates]
            )
            single = np.split(
                _single_faces(keys),
                np.cumsum([len(group["cells"]) for group, ring in candidates])[:-1],
            )
            for (group, ring), mask in zip(candidates, single):
                faces.append((group, ring, np.flatnonzero(mask)))

    for group in groups:
        if group["kind"] in RINGS:
            index = np.arange(len(group["cells"]))
            faces.append((group, RINGS[group["kind"]], index))

    templates = {size: _template(size, n) for size in [2, 3, 4]}

    # the corner points of the surface
    corners = [group["cells"][index][:, ring] for group, ring, index in faces]
    corner_points, corner_index = np.unique(
        np.concatenate([c.ravel() for c in corners] or [np.zeros(0, np.int64)]),
        return_inverse=True,
    )
    corner_index = np.split(corner_index, np.cumsum([c.size for c in corners])[:-1])
    corner_index = [ci.reshape(c.shape) for ci, c in zip(corner_index, corners)]

    # the edges of the surface, numbered by their sorted corner points
    occurrences = []
    for (group, ring, index), c in zip(faces, corner_index):
        template = templates[len(ring)]
        for e, (a, b) in enumerate(template.edges):
            occurrences.append((c[:, a], c[:, b]))

    starts = np.concatenate([a for a, b in occurrences] or [np.zeros(0, np.int64)])
    ends = np.concatenate([b for a, b in occurrences] or [np.zeros(0, np.int64)])
    lower, upper = np.minimum(starts, ends), np.maximum(starts, ends)
    edge_keys = lower * max(len(corner_points), 1) + upper
    unique_keys, first, edge_index = np.unique(
        edge_keys, return_index=True, return_inverse=True
    )
    nedges = len(unique_keys)
    forward = starts < ends

    # the numbering of the points: corners, points of the edges, interior points
    nedge_points = n - 1
    edge_offset = len(corner_points)
    interior_offset = edge_offset + nedges * nedge_points

    rows, cols, values = [], [], []

    # the corner points are the points of the meshes
    rows.append(np.arange(len(corner_points)))
    cols.append(corner_points)
    values.append(np.ones(len(corner_points)))

    triangles, owners, face_numbers = [], [], []
    face_offset = 0
    is_first = np.zeros(len(edge_keys), dtype=bool)
    is_first[first] = True
    occurrence = 0

    for (group, ring, index), c in zip(faces, corner_index):
        template = templates[len(ring)]
        cells = group["cells"][index]
        weights = template.weights(ring, group["element"], cells.shape[1])
        ninstances = len(index)

        # the points of the surface of the points of the template of each instance
        points = np.zeros((ninstances, len(template.params)), dtype=np.int64)
        points[:, template.corners >= 0] = c[:, template.corners[template.corners >= 0]]

        for e in range(len(template.edges)):
            occurrences_e = slice(occurrence, occurrence + ninstances)
            edges_e = edge_index[occurrences_e]
            forward_e = forward[occurrences_e]
            position = np.arange(nedge_points)
            local = np.where(forward_e[:, None], position, nedge_points - 1 - position)
            global_points = edge_offset + edges_e[:, None] * nedge_points + local
            points[:, template.edge_points[e]] = global_points

            # the weights of the points of an edge are taken from its first face
            first_e = is_first[occurrences_e]
            if nedge_points > 0 and np.any(first_e):
                _append_rows(
                    rows,
                    cols,
                    values,
                    global_points[first_e],
                    cells[first_e],
                    weights[template.edge_points[e]],
                )

            occurrence += ninstances

        ninterior = len(template.interior)
        if ninterior > 0:
            interior = interior_offset + np.arange(ninstances * ninterior).reshape(
                ninstances, ninterior
            )
            points[:, template.interior] = interior
            _append_rows(
                rows, cols, values, interior, cells, weights[template.interior]
            )
            interior_offset += ninstances * ninterior

        if len(template.triangles) > 0:
            triangles.append(points[:, template.triangles].reshape(-1, 3))
            owners.append(
                np.repeat(group["cell_offset"] + index, len(template.triangles))
            )
            face_numbers.append(
                np.repeat(face_offset + np.arange(ninstances), len(template.triangles))
            )
            face_offset += ninstances

    # the line segments of the edges, from the lower to the upper corner point
    segments = np.zeros((nedges, n + 1), dtype=np.int64)
    segments[:, 0] = lower[first]
    segments[:, -1] = upper[first]
    segments[:, 1:-1] = edge_offset + np.arange(nedges * nedge_points).reshape(
        nedges, nedge_points
    )
    edges = np.stack([segments[:, :-1], segments[:, 1:]], axis=-1).reshape(-1, 2)

    rows = np.concatenate(rows)
    interpolation = csr_matrix(
        (np.concatenate(values), (rows, np.concatenate(cols))),
        shape=(interior_offset, total_points),
    )

    return Surface(
        interpolation=interpolation,
        triangles=np.concatenate(triangles or [np.zeros((0, 3), np.int64)]),
        owners=np.concatenate(owners or [np.zeros(0, np.int64)]),
        faces=np.concatenate(face_numbers or [np.zeros(0, np.int64)]),
        edges=edges,
    )


def _append_rows(rows, cols, values, points, cells, weights):
    """Append the rows of the interpolation matrix of points (ninstances, k) with the
    weights (k, points_per_cell) of the points of the cells (ninstances, nodes)."""

    k, nz = np.nonzero(weights)
    rows.append(np.repeat(points, 0) if points.size == 0 else points[:, k].ravel())
    cols.append(cells[:, nz].ravel())
    values.append(np.broadcast_to(weights[k, nz], (len(points), len(k))).ravel())

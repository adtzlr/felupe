# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract the outer surface of a FElupe mesh for the viewport of the dashboard."""

import numpy as np

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
}


def _unique_faces(faces):
    "Return the faces which belong to only one cell (with their original ordering)."
    if len(faces) == 0:
        return faces
    keys = np.sort(faces, axis=1)
    _, index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[np.sort(index[counts == 1])]


def surface(meshes):
    """Return the triangles, quads and lines of the outer surface of one or more
    meshes with a common points-array.

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

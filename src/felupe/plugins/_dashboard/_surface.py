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


def surface(mesh):
    """Return the triangles, quads and lines of the outer surface of a mesh.

    Parameters
    ----------
    mesh : felupe.Mesh
        The mesh with points, cells and a cell type.

    Returns
    -------
    dict
        A dict with the arrays ``"tri"`` (n, 3), ``"quad"`` (n, 4) and ``"line"``
        (n, 2) of point indices and the name of the base cell type ``"kind"``.
    """

    cells = np.asarray(mesh.cells)
    kind, ncorners = CORNERS.get(mesh.cell_type, (None, None))

    tri = np.zeros((0, 3), dtype=np.int64)
    quad = np.zeros((0, 4), dtype=np.int64)
    line = np.zeros((0, 2), dtype=np.int64)

    if kind is None:
        # unknown cell type: draw the points only
        return dict(kind="points", tri=tri, quad=quad, line=line)

    corners = cells[:, :ncorners]

    if kind in FACES:
        faces3 = [f for f in FACES[kind] if len(f) == 3]
        faces4 = [f for f in FACES[kind] if len(f) == 4]
        if faces3:
            tri = _unique_faces(corners[:, faces3].reshape(-1, 3))
        if faces4:
            quad = _unique_faces(corners[:, faces4].reshape(-1, 4))
    elif kind == "quad":
        quad = corners
    elif kind == "triangle":
        tri = corners
    elif kind == "line":
        line = corners

    return dict(kind=kind, tri=tri, quad=quad, line=line)

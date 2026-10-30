# SPDX-License-Identifier: GPL-3.0-or-later

from ._container import MeshContainer
from ._convert import (
    add_midpoints_edges,
    add_midpoints_faces,
    add_midpoints_volumes,
    cell_types,
    collect_edges,
    collect_faces,
    collect_volumes,
    convert,
    subdivide,
)
from ._dual import dual
from ._geometry import (
    Circle,
    Cube,
    CubeArbitraryOrderHexahedron,
    Grid,
    Line,
    Point,
    Rectangle,
    RectangleArbitraryOrderQuad,
    Triangle,
)
from ._interpolate import interpolate_line
from ._mesh import Mesh
from ._read import read
from ._tools import (
    concatenate,
    expand,
    extrude,
    fill_between,
    flip,
    merge_duplicate_cells,
    merge_duplicate_points,
    mirror,
    revolve,
    rotate,
    runouts,
    stack,
    translate,
    triangulate,
)
from ._tools import merge_duplicate_points as sweep

__all__ = [
    "MeshContainer",
    "add_midpoints_edges",
    "add_midpoints_faces",
    "add_midpoints_volumes",
    "cell_types",
    "collect_edges",
    "collect_faces",
    "collect_volumes",
    "convert",
    "dual",
    "Circle",
    "Cube",
    "CubeArbitraryOrderHexahedron",
    "Grid",
    "Line",
    "Point",
    "Rectangle",
    "RectangleArbitraryOrderQuad",
    "Triangle",
    "Mesh",
    "read",
    "concatenate",
    "expand",
    "extrude",
    "flip",
    "fill_between",
    "interpolate_line",
    "mirror",
    "merge_duplicate_points",
    "merge_duplicate_cells",
    "revolve",
    "rotate",
    "runouts",
    "stack",
    "subdivide",
    "sweep",
    "translate",
    "triangulate",
]

r"""
Script-based Hex-meshing
------------------------

.. topic:: Create a 3d dynamic mesh for a metacone component out of hexahedrons.

   * apply :ref:`mesh-tools <felupe-api-mesh>`

   * create a :class:`~felupe.MeshContainer` for meshes associated to two materials
"""

import numpy as np

import felupe as fem

layers = [2, 11, 2, 11, 2]
lines = [
    fem.mesh.Line(a=0, b=13, n=21).expand(n=1).translate(2, axis=1),
    fem.mesh.Line(a=0, b=13, n=21).expand(n=1).translate(2.5, axis=1),
    fem.mesh.Line(a=-0.2, b=10, n=21).expand(n=1).translate(4.5, axis=1),
    fem.mesh.Line(a=-0.2, b=10, n=21).expand(n=1).translate(5, axis=1),
    fem.mesh.Line(a=-0.4, b=7, n=21).expand(n=1).translate(6.5, axis=1),
    fem.mesh.Line(a=-0.4, b=7, n=21).expand(n=1).translate(7, axis=1),
]
faces = fem.MeshContainer(
    [
        first.fill_between(second, n=n)
        for first, second, n in zip(lines[:-1], lines[1:], layers)
    ]
)
point = lambda m: m.points[np.unique(m.cells)].mean(axis=0) - np.array([2, 0])
mask = lambda m: np.unique(m.cells)
kwargs = dict(axis=1, exponent=2.5, normalize=True)
faces.points[:] = (
    faces[1]
    .add_runouts([3], centerpoint=point(faces[1]), mask=mask(faces[1]), **kwargs)
    .points
)
faces.points[:] = (
    faces[3]
    .add_runouts([7], centerpoint=point(faces[3]), mask=mask(faces[3]), **kwargs)
    .points
)
faces.points[:21] = faces.points[21 : 2 * 21]
faces.points[-21:] = faces.points[-2 * 21 : -21]
faces.points[:] = faces[0].rotate(15, axis=2).points
faces[0].y[:21] = 2
faces[-1].y[-21:] = 8.5
mesh = fem.MeshContainer([faces.stack([1, 3]), faces.stack([0, 2, 4])])

mesh_3d = fem.MeshContainer(
    [m.revolve(n=73, phi=270).rotate(-90, axis=1).rotate(-90, axis=2) for m in mesh]
)
mesh_3d.plot(colors=[None, "white"], show_edges=True, opacity=1).show()

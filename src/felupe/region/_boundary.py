# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from ..math import cross
from ..mesh import Mesh
from ._region import Region


def boundary_cells_quad(mesh):
    "Convert the cells array of a quad mesh into a boundary cells array."

    # edges (boundary) of a quad
    i = [3, 1, 0, 2]
    j = [0, 2, 1, 3]

    cells_faces = np.dstack(
        (
            mesh.cells[:, i],
            mesh.cells[:, j],
        )
    )

    # complementary edges for the creation of "boundary" quads
    # (rotated quads with 1st edge as n-th edge of one original quad)
    t = [1, 0, 3, 2]
    k = np.array(i)[t]
    m = np.array(j)[t]

    cells = np.dstack(
        (
            cells_faces,
            mesh.cells[:, k],
            mesh.cells[:, m],
        )
    )

    return cells, cells_faces


def boundary_cells_quad8(mesh):
    "Convert the cells array of a quadratic quad mesh into a boundary cells array."

    cells_quad, cells_faces_quad = boundary_cells_quad(mesh)

    # midpoints on edges (boundary) of a quadratic quad
    m = [7, 5, 4, 6]

    cells_faces = np.dstack(
        (
            cells_faces_quad,
            mesh.cells[:, m],
        )
    )

    # complementary midpoints of edges for the creation of "boundary" quadratic quads
    # (rotated quads with 1st edge as n-th edge of one original quad)
    n = [5, 6, 7, 4]
    p = [4, 5, 6, 7]
    q = [6, 7, 4, 5]

    cells = np.dstack(
        (
            cells_quad,
            mesh.cells[:, m],
            mesh.cells[:, n],
            mesh.cells[:, p],
            mesh.cells[:, q],
        )
    )

    return cells, cells_faces


def boundary_cells_quad9(mesh):
    "Convert the cells array of a bi-quadratic quad mesh into a boundary cells array."

    cells_quad8, cells_faces_quad8 = boundary_cells_quad8(mesh)

    # midpoints on edges (boundary) of a quadratic quad
    r = [8, 8, 8, 8]

    cells = np.dstack(
        (
            cells_quad8,
            mesh.cells[:, r],
        )
    )

    return cells, cells_faces_quad8


def boundary_cells_hexahedron(mesh):
    "Convert the cells array of a hex mesh into a boundary cells array."

    # faces (boundary) of a hexahedron
    i = [0, 2, 1, 3, 0, 5]
    j = [3, 1, 0, 2, 1, 4]
    k = [7, 5, 4, 6, 2, 7]
    r = [4, 6, 5, 7, 3, 6]

    cells_faces = np.dstack(
        (
            mesh.cells[:, i],
            mesh.cells[:, j],
            mesh.cells[:, k],
            mesh.cells[:, r],
        )
    )

    # complementary faces for the creation of "boundary" hexahedrons
    # (6 rotated hexahedrons with 1st face as n-th face of
    #  one original hexahedron)
    m = [1, 3, 2, 0, 4, 1]
    n = [2, 0, 3, 1, 5, 0]
    p = [6, 4, 7, 5, 6, 3]
    q = [5, 7, 6, 4, 7, 2]

    cells = np.dstack(
        (
            cells_faces,
            mesh.cells[:, m],
            mesh.cells[:, n],
            mesh.cells[:, p],
            mesh.cells[:, q],
        )
    )

    return cells, cells_faces


def boundary_cells_hexahedron20(mesh):
    "Convert the cells array of a quadratic hex mesh into a boundary cells array."

    cells_hexahedron, cells_faces_hexahedron = boundary_cells_hexahedron(mesh)

    # midpoints on edges of faces (boundary) of a hexahedron
    i = [11, 9, 8, 10, 8, 12]
    j = [19, 17, 16, 18, 9, 15]
    k = [15, 13, 12, 14, 10, 14]
    v = [16, 18, 17, 19, 11, 13]

    cells_faces = np.dstack(
        (
            cells_faces_hexahedron,
            mesh.cells[:, i],
            mesh.cells[:, j],
            mesh.cells[:, k],
            mesh.cells[:, v],
        )
    )

    # complementary faces for the creation of "boundary" hexahedrons
    # (6 rotated hexahedrons with 1st face as n-th face of
    #  one original hexahedron)
    m = [9, 11, 10, 8, 12, 8]
    n = [18, 16, 19, 17, 13, 11]
    p = [13, 15, 14, 12, 14, 10]
    q = [17, 19, 18, 16, 15, 9]

    r = [8, 10, 9, 11, 16, 17]
    s = [10, 8, 11, 9, 17, 16]
    t = [14, 12, 15, 13, 18, 19]
    u = [12, 14, 13, 15, 19, 18]

    cells = np.dstack(
        (
            cells_hexahedron,
            mesh.cells[:, i],
            mesh.cells[:, j],
            mesh.cells[:, k],
            mesh.cells[:, v],
            mesh.cells[:, m],
            mesh.cells[:, n],
            mesh.cells[:, p],
            mesh.cells[:, q],
            mesh.cells[:, r],
            mesh.cells[:, s],
            mesh.cells[:, t],
            mesh.cells[:, u],
        )
    )

    return cells, cells_faces


def boundary_cells_hexahedron27(mesh):
    "Convert the cells array of a bi-quadratic hex mesh into a boundary cells array."

    cells_hexahedron20, cells_faces_hexahedron20 = boundary_cells_hexahedron20(mesh)

    # midpoints of faces (boundary) of a hexahedron
    i = [20, 21, 22, 23, 24, 25]

    cells_faces = np.dstack(
        (
            cells_faces_hexahedron20,
            mesh.cells[:, i],
        )
    )

    # complementary faces for the creation of "boundary" hexahedrons
    # (6 rotated hexahedrons with 1st face as n-th face of
    #  one original hexahedron)
    j = [22, 23, 21, 20, 20, 21]
    k = [23, 22, 20, 21, 21, 20]
    r = [24, 25, 24, 25, 22, 23]
    m = [25, 24, 25, 24, 23, 22]
    n = [20, 21, 22, 23, 24, 25]
    p = [21, 20, 23, 22, 25, 24]
    q = [26, 26, 26, 26, 26, 26]

    cells = np.dstack(
        (
            cells_hexahedron20,
            mesh.cells[:, j],
            mesh.cells[:, k],
            mesh.cells[:, r],
            mesh.cells[:, m],
            mesh.cells[:, n],
            mesh.cells[:, p],
            mesh.cells[:, q],
        )
    )

    return cells, cells_faces


class RegionBoundary(Region):
    r"""
    A numeric boundary-region as a combination of a mesh, an element and a
    numeric integration scheme (quadrature rule).

    Parameters
    ----------
    mesh : Mesh
        A mesh with points and cells.
    element : Element
        The finite element formulation to be applied on the cells.
    quadrature: Quadrature
        An element-compatible numeric integration scheme with points and weights.
    grad : bool, optional
        A flag to invoke gradient evaluation (default is True).
    only_surface: bool, optional
        A flag to use only the enclosing outline of the region (default is True).
    mask: ndarray or None, optional
        A boolean array to select a specific set of points (default is None).
    ensure_3d : bool, optional
        A flag to enforce 3d area normal vectors.

    Attributes
    ----------
    mesh : Mesh
        A mesh with points and cells.
    element : Finite element
        The finite element formulation to be applied on the cells.
    quadrature: Quadrature scheme
        An element-compatible numeric integration scheme with points and weights.
    h : ndarray
        Element shape function array ``h_aq`` of shape function ``a`` evaluated at
        quadrature point ``q``.
    dhdr : ndarray
        Partial derivative of element shape function array ``dhdr_aJq`` with shape
        function ``a`` w.r.t. natural element coordinate ``J`` evaluated at quadrature
        point ``q`` for every cell ``c`` (geometric gradient or **Jacobian**
        transformation between ``X`` and ``r``).
    dXdr : ndarray
        Geometric gradient ``dXdr_IJqc`` as partial derivative of undeformed coordinate
        ``I`` w.r.t. natural element coordinate ``J`` evaluated at quadrature point
        ``q`` for every cell ``c`` (geometric gradient or **Jacobian** transformation
        between ``X`` and ``r``).
    drdX : ndarray
        Inverse of dXdr.
    dA : ndarray
        Numeric *Differential area vectors*.
    normals : ndarray
        Area unit normal vectors.
    dV : ndarray
        Numeric *Differential volume element* as norm of *Differential area vectors*.
    dhdX : ndarray
        Partial derivative of element shape functions ``dhdX_aJqc`` of shape function
        ``a`` w.r.t. undeformed coordinate ``J`` evaluated at quadrature point ``q`` for
        every cell ``c``.

    Notes
    -----
    The gradients of the element shape functions w.r.t the undeformed coordinates are
    evaluated at all integration points of each cell in the region if the optional
    gradient argument is ``True``.

    .. math::

       \frac{\partial X_I}{\partial r_J} &= \hat{X}_{aI} \frac{
               \partial h_a}{\partial r_J
           }

       \frac{\partial h_a}{\partial X_J} &= \frac{\partial h_a}{\partial r_I}
       \frac{\partial r_I}{\partial X_J}

       dV &= \det\left(\frac{\partial X_I}{\partial r_J}\right) w

    Examples
    --------
    >>> import felupe as fem

    >>> mesh = fem.Rectangle(n=(3, 2))
    >>> element = fem.Quad()
    >>> quadrature = fem.GaussLegendreBoundary(order=1, dim=2)

    >>> region = fem.RegionBoundary(mesh, element, quadrature)
    >>> region
    <felupe Region object>
      Element formulation: Quad
      Quadrature rule: GaussLegendreBoundary
      Gradient evaluated: True

    The numeric differential area vectors are the products of the cofactors of the
    geometric gradient :math:`\partial X_I / \partial r_J` and the weights `w` of the
    quadrature points. The differential area vectors array is of shape
    ``(nquadraturepoints, ndim, nboundarycells)``.

    >>> region.dA
    array([[[ 0.  , -0.5 ,  0.  ,  0.5 ,  0.  ,  0.  ],
            [ 0.  , -0.5 ,  0.  ,  0.5 ,  0.  ,  0.  ]],
    <BLANKLINE>
           [[-0.25, -0.  , -0.25, -0.  ,  0.25,  0.25],
            [-0.25, -0.  , -0.25, -0.  ,  0.25,  0.25]]])

    In a boundary region, the numeric differential volumes are the magnitudes of the
    differential area vectors. For a quad mesh, the boundary cell volumes are the edge
    lengths.

    >>> region.dV.sum(axis=0)
    array([0.5, 1. , 0.5, 1. , 0.5, 0.5])

    Unit normal vectors are obtained by the ratio of the differential area vectors and
    the differential volumes.

    >>> region.dA / region.dV  ## this is equal to ``region.normals``
    array([[[ 0., -1.,  0.,  1.,  0.,  0.],
            [ 0., -1.,  0.,  1.,  0.,  0.]],
    <BLANKLINE>
           [[-1., -0., -1., -0.,  1.,  1.],
            [-1., -0., -1., -0.,  1.,  1.]]])

    The partial derivative of the first element shape function w.r.t. the undeformed
    coordinates evaluated at the second integration point of the last element of the
    region:

    >>> region.dhdX[0, :, 1, -1]
    array([2.        , 0.21132487])
    """

    def __init__(
        self,
        mesh,
        element,
        quadrature,
        grad=True,
        only_surface=True,
        mask=None,
        ensure_3d=False,
    ):
        self.only_surface = only_surface
        self.mask = mask
        self.ensure_3d = ensure_3d

        if mesh.cell_type == "quad":
            cells, cells_faces = boundary_cells_quad(mesh)
        elif mesh.cell_type == "quad8":
            cells, cells_faces = boundary_cells_quad8(mesh)
        elif mesh.cell_type == "quad9":
            cells, cells_faces = boundary_cells_quad9(mesh)
        elif mesh.cell_type == "hexahedron":
            cells, cells_faces = boundary_cells_hexahedron(mesh)
        elif mesh.cell_type == "hexahedron20":
            cells, cells_faces = boundary_cells_hexahedron20(mesh)
        elif mesh.cell_type == "hexahedron27":
            cells, cells_faces = boundary_cells_hexahedron27(mesh)
        else:
            raise NotImplementedError("Cell type not supported.")

        cells_faces = cells_faces.reshape(-1, cells_faces.shape[-1])
        cells = cells.reshape(-1, cells.shape[-1])

        if self.only_surface:
            # sort faces, get indices of unique faces and counts
            cells_faces_sorted = np.sort(cells_faces, axis=1)
            cells_faces_unique, index, counts = np.unique(
                cells_faces_sorted, True, False, True, 0
            )

            self._index = index
            self._mask = counts == 1
        else:
            self._index = np.arange(len(cells_faces))
            self._mask = np.ones_like(self._index, dtype=bool)

        self._selection = self._index[self._mask]

        # merge with point-mask
        if mask is not None:
            point_selection = np.arange(len(mesh.points))[mask]
            self._selection = self._selection[
                np.all(np.isin(cells_faces[self._selection], point_selection), axis=1)
            ]

        # get cell-faces and cells on boundary (unique cell-faces with one count)
        cells_on_boundary = cells[self._selection]

        # create mesh on boundary
        mesh_boundary_cells = mesh.copy()
        mesh_boundary_cells.update(cells=cells_on_boundary)
        self.mesh = mesh_boundary_cells
        self.mesh.cells_faces = cells_faces[self._selection]

        # init region and faces
        super().__init__(mesh_boundary_cells, element, quadrature, grad=grad)

        if grad:
            self.dA, self.dV, self.normals = self._init_faces()

    def _init_faces(self):
        "Initialize (norm of) face normals of cells."

        if (
            self.mesh.cell_type == "quad"
            or self.mesh.cell_type == "quad8"
            or self.mesh.cell_type == "quad9"
        ):
            dA_1 = self.dXdr[:, 0][::-1]
            dA_1[0] = -dA_1[0]

        elif (
            self.mesh.cell_type == "hexahedron"
            or self.mesh.cell_type == "hexahedron20"
            or self.mesh.cell_type == "hexahedron27"
        ):
            dA_1 = cross(self.dXdr[:, 0], self.dXdr[:, 1])

        dA = -dA_1 * self.quadrature.weights.reshape(-1, 1)

        # norm and unit normal vector
        dV = np.linalg.norm(dA, axis=0)
        normals = dA / dV

        if self.ensure_3d:
            if dA.shape[0] == 2:
                dA = np.pad(dA, ((0, 1), (0, 0), (0, 0)))
                normals = np.pad(normals, ((0, 1), (0, 0), (0, 0)))

        return dA, dV, normals

    def mesh_faces(self):
        "Return a Mesh with face-cells on the selected boundary."

        face_type = {
            "quad": "line",
            "hexahedron": "quad",
            "quad8": "line3",
            "quad9": "line3",
        }[self.mesh.cell_type]
        return Mesh(self.mesh.points, self.mesh.cells_faces, face_type)

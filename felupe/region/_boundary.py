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

from ..math import cross
from ._region import Region
from ..mesh import Mesh


class RegionBoundary(Region):
    r"""
    A numeric boundary-region as a combination of a mesh, an element and a
    numeric integration scheme (quadrature). The gradients of the element shape
    functions are evaluated at all integration points of each cell in the region
    if the optional gradient argument is True.

    .. math::

       \frac{\partial X^I}{\partial r^J} &= X_a^I \frac{\partial h_a}{\partial r^J}

       \frac{\partial h_a}{\partial X^J} &= \frac{\partial h_a}{\partial r^I} \frac{\partial r^I}{\partial X^J}

       dV &= \det\left(\frac{\partial X^I}{\partial r^J}\right) w


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
        Element shape function array ``h_ap`` of shape function ``a`` evaluated at quadrature point ``p``.
    dhdr : ndarray
        Partial derivative of element shape function array ``dhdr_aJp`` with shape function ``a`` w.r.t. natural element coordinate ``J`` evaluated at quadrature point ``p`` for every cell ``c`` (geometric gradient or **Jacobian** transformation between ``X`` and ``r``).
    dXdr : ndarray
        Geometric gradient ``dXdr_IJpc`` as partial derivative of undeformed coordinate ``I`` w.r.t. natural element coordinate ``J`` evaluated at quadrature point ``p`` for every cell ``c`` (geometric gradient or **Jacobian** transformation between ``X`` and ``r``).
    drdX : ndarray
        Inverse of dXdr.
    dA : ndarray
        Numeric *Differential area vectors*.
    normals : ndarray
        Area unit normal vectors.
    dV : ndarray
        Numeric *Differential volume element* as norm of *Differential area vectors*.
    dhdX : ndarray
        Partial derivative of element shape functions ``dhdX_aJpc`` of shape function ``a`` w.r.t. undeformed coordinate ``J`` evaluated at quadrature point ``p`` for every cell ``c``.
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
            l = np.array(j)[t]

            cells = np.dstack(
                (
                    mesh.cells[:, i],
                    mesh.cells[:, j],
                    mesh.cells[:, k],
                    mesh.cells[:, l],
                )
            )

        elif mesh.cell_type == "hexahedron":

            # faces (boundary) of a hexahedron
            i = [0, 1, 1, 2, 0, 4]
            j = [3, 2, 0, 3, 1, 5]
            k = [7, 6, 4, 7, 2, 6]
            l = [4, 5, 5, 6, 3, 7]

            cells_faces = np.dstack(
                (
                    mesh.cells[:, i],
                    mesh.cells[:, j],
                    mesh.cells[:, k],
                    mesh.cells[:, l],
                )
            )

            # complementary faces for the creation of "boundary" hexahedrons
            # (6 rotated hexahedrons with 1st face as n-th face of
            #  one original hexahedron)
            t = [1, 0, 3, 2, 5, 4]
            m = np.array(i)[t]
            n = np.array(j)[t]
            p = np.array(k)[t]
            q = np.array(l)[t]

            cells = np.dstack(
                (
                    mesh.cells[:, i],
                    mesh.cells[:, j],
                    mesh.cells[:, k],
                    mesh.cells[:, l],
                    mesh.cells[:, m],
                    mesh.cells[:, n],
                    mesh.cells[:, p],
                    mesh.cells[:, q],
                )
            )
            # ensure right-hand-side cell connectivity
            for a in [1, 3, 5]:
                cells[:, a, :4] = cells[:, a, :4].T[::-1].T
                cells[:, a, 4:] = cells[:, a, 4:].T[::-1].T

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

        ## create mesh on boundary
        mesh_boundary_cells = mesh.copy()
        mesh_boundary_cells.update(cells_on_boundary)
        self.mesh = mesh_boundary_cells
        self.mesh.cells_faces = cells_faces[self._selection]

        # init region and faces
        super().__init__(mesh_boundary_cells, element, quadrature, grad=grad)

        if grad:
            self.dA, self.dV, self.normals = self._init_faces()

    def _init_faces(self):
        "Initialize (norm of) face normals of cells."

        if self.mesh.cell_type == "quad":

            dA_1 = self.dXdr[:, 0][::-1]
            dA_1[0] = -dA_1[0]

        elif self.mesh.cell_type == "hexahedron":
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

        face_type = {"quad": "line", "hexahedron": "quad"}[self.mesh.cell_type]
        return Mesh(self.mesh.points, self.mesh.cells_faces, face_type)

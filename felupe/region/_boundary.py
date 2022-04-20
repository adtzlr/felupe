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

        # number of faces
        self.nfaces = quadrature.nfaces

        if mesh.cell_type == "quad":

            # edges (generalized "faces") of a quad
            i = [3, 1, 0, 2]
            j = [0, 2, 1, 3]

            faces = np.dstack(
                (
                    mesh.cells[:, i],
                    mesh.cells[:, j],
                )
            )
            faces = faces.reshape(-1, faces.shape[-1])

            # complementary edges for quad faces (generalized "volumes")
            t = [1, 0, 3, 2]
            k = np.array(i)[t]
            l = np.array(j)[t]

            volumes = np.dstack(
                (
                    mesh.cells[:, i],
                    mesh.cells[:, j],
                    mesh.cells[:, k],
                    mesh.cells[:, l],
                )
            )

        elif mesh.cell_type == "hexahedron":

            # faces of hexahedron
            i = [0, 1, 1, 2, 0, 4]
            j = [3, 2, 0, 3, 1, 5]
            k = [7, 6, 4, 7, 2, 6]
            l = [4, 5, 5, 6, 3, 7]

            faces = np.dstack(
                (
                    mesh.cells[:, i],
                    mesh.cells[:, j],
                    mesh.cells[:, k],
                    mesh.cells[:, l],
                )
            )

            # complementary faces for hexahedron volumes
            t = [1, 0, 3, 2, 5, 4]
            m = np.array(i)[t]
            n = np.array(j)[t]
            p = np.array(k)[t]
            q = np.array(l)[t]

            volumes = np.dstack(
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

        else:
            raise NotImplementedError("Cell type not supported.")

        faces = faces.reshape(-1, faces.shape[-1])
        volumes = volumes.reshape(-1, volumes.shape[-1])
        self._faces = faces

        if self.only_surface:
            # sort faces, get indices of unique faces and counts
            faces_sorted = np.sort(faces, axis=1)
            faces_unique, index, counts = np.unique(faces_sorted, True, False, True, 0)

            self._index = index
            self._mask = counts == 1
        else:
            self._index = np.arange(len(faces))
            self._mask = np.ones_like(self._index, dtype=bool)

        self._selection = self._index[self._mask]

        # merge with point-mask
        if mask is not None:
            point_selection = np.arange(len(mesh.points))[mask]
            self._selection = self._selection[
                np.all(np.isin(self._faces[self._selection], point_selection), axis=1)
            ]

        # get faces and volumes on boundary (unique faces with one count)
        volumes_boundary = volumes[self._selection]

        # init region and faces
        super().__init__(mesh, element, quadrature, grad=grad)

        # reshape data
        self.h = self._reshape(self.h)

        if grad:
            self.dA, self.dV, self.normals = self._init_faces()
            
            self.dA = self._reshape(self.dA)
            self.dV = self._reshape(self.dV)
            self.normals = self._reshape(self.normals)

            self.dhdr = self._reshape(self.dhdr)
            self.dhdX = self._reshape(self.dhdX)
            self.dXdr = self._reshape(self.dXdr)

        ## create mesh on boundary
        mesh_boundary = mesh.copy()
        mesh_boundary.update(volumes_boundary)
        self.mesh = mesh_boundary

    def _reshape(self, A):
        "Reshape data."

        # reshape functions for all boundary cells
        B = A.reshape(*A.shape[:-2], self.nfaces, -1, A.shape[-1])
        a = np.arange(len(B.shape))

        a[-3:] = a[[-2, -3, -1]]
        C = B.transpose(a)
        D = C.reshape(*A.shape[:-2], -1, self.nfaces * A.shape[-1])

        # slice faces on boundary
        return D.T[self._selection].T

    def _init_faces(self):
        "Initialize (norm of) face normals of cells."

        if self.mesh.cell_type == "quad":
            dA_1 = self.dXdr[:, 1][::-1]
            dA_2 = self.dXdr[:, 0][::-1]
            dA_faces = np.array([-dA_1, dA_1, -dA_2, dA_2])

        elif self.mesh.cell_type == "hexahedron":
            dA_1 = cross(self.dXdr[:, 1], self.dXdr[:, 2])
            dA_2 = cross(self.dXdr[:, 2], self.dXdr[:, 0])
            dA_3 = cross(self.dXdr[:, 0], self.dXdr[:, 1])
            dA_faces = np.array(
                [
                    -dA_1,
                    dA_1,
                    -dA_2,
                    dA_2,
                    -dA_3,
                    dA_3,
                ]
            )

        dA = np.stack(dA_faces) * self.quadrature.weights.reshape(-1, 1)
        nfaces, dim, nqpoints, ncells = dA.shape
        nqpoints_per_face = nqpoints // nfaces
        dA = dA.reshape(nfaces, dim, nfaces, nqpoints_per_face, ncells)
        dA = np.einsum("ijk...->ikj...", dA)
        dA = np.array([dA[a, a] for a in range(dA.shape[0])])
        dA = np.einsum("ijkl...->jikl...", dA)
        dA = dA.reshape(dim, nqpoints, ncells)
        dV = np.linalg.norm(dA, axis=0)
        normals = dA / dV

        if self.ensure_3d:

            if dA.shape[0] == 2:
                dA = np.pad(dA, ((0, 1), (0, 0), (0, 0)))
                normals = np.pad(normals, ((0, 1), (0, 0), (0, 0)))

        return dA, dV, normals

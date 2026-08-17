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
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from ..assembly import IntegralForm
from ._helpers import Assemble, Results

# second derivative of the shape functions of a bilinear quad w.r.t. the natural
# element coordinates: only the mixed derivative is non-zero
D2HDRDS = np.array([0.25, -0.25, 0.25, -0.25])


def shape_function_quad(coordinates):
    r"""Return the shape functions and their gradients of a bilinear quad, evaluated at
    given natural element coordinates.

    Parameters
    ----------
    coordinates : ndarray of shape (..., 2)
        The natural element coordinates :math:`(r, s)`.

    Returns
    -------
    h : ndarray of shape (..., 4)
        The shape functions :math:`h_a`.
    dhdr : ndarray of shape (..., 4, 2)
        The gradients of the shape functions :math:`\partial h_a / \partial r_J`.
    """

    r, s = coordinates[..., 0], coordinates[..., 1]
    ar, as_ = (1 - r) / 4, (1 - s) / 4
    br, bs = (1 + r) / 4, (1 + s) / 4

    h = np.empty((*coordinates.shape[:-1], 4))
    h[..., 0] = 4 * ar * as_
    h[..., 1] = 4 * br * as_
    h[..., 2] = 4 * br * bs
    h[..., 3] = 4 * ar * bs

    dhdr = np.empty((*coordinates.shape[:-1], 4, 2))
    dhdr[..., 0, 0], dhdr[..., 0, 1] = -as_, -ar
    dhdr[..., 1, 0], dhdr[..., 1, 1] = as_, -br
    dhdr[..., 2, 0], dhdr[..., 2, 1] = bs, br
    dhdr[..., 3, 0], dhdr[..., 3, 1] = -bs, ar

    return h, dhdr


def solve_2x2(matrix, vector):
    "Return the solution of a batch of 2x2 linear equation systems."

    determinant = (
        matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    )

    # regularize singular systems, the affected pairs are filtered out later on
    small = np.abs(determinant) < np.finfo(float).tiny
    determinant = np.where(small, 1.0, determinant)

    solution = np.stack(
        [
            matrix[..., 1, 1] * vector[..., 0] - matrix[..., 0, 1] * vector[..., 1],
            matrix[..., 0, 0] * vector[..., 1] - matrix[..., 1, 0] * vector[..., 0],
        ],
        axis=-1,
    )

    return solution / determinant[..., None]


def invert_2x2(matrix):
    "Return the inverse of a batch of symmetric 2x2 matrices."

    determinant = (
        matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    )
    small = np.abs(determinant) < np.finfo(float).tiny
    determinant = np.where(small, 1.0, determinant)

    inverse = np.stack(
        [
            np.stack([matrix[..., 1, 1], -matrix[..., 0, 1]], axis=-1),
            np.stack([-matrix[..., 1, 0], matrix[..., 0, 0]], axis=-1),
        ],
        axis=-2,
    )

    return inverse / determinant[..., None, None]


def closest_point_projection(points, vertices, maxiter=12, tol=1e-10):
    r"""Return the natural element coordinates of the closest-point projections of
    points onto bilinear quad faces.

    Parameters
    ----------
    points : ndarray of shape (npairs, 3)
        The coordinates of the points to be projected.
    vertices : ndarray of shape (npairs, 4, 3)
        The coordinates of the vertices of the quad faces.
    maxiter : int, optional
        The maximum number of local Newton iterations (default is 12).
    tol : float, optional
        The tolerance for the local Newton iterations (default is 1e-10).

    Returns
    -------
    coordinates : ndarray of shape (npairs, 2)
        The natural element coordinates of the projected points.
    converged : ndarray of shape (npairs,)
        A mask with the pairs for which the projection converged.

    Notes
    -----
    The projection minimizes the squared distance between a point and a face, see Eq.
    :eq:`closest-point`.

    ..  math::
        :label: closest-point

        f(\boldsymbol{\xi}) = \frac{1}{2} \left(
            \boldsymbol{x} - \hat{\boldsymbol{x}}(\boldsymbol{\xi})
        \right) \cdot \left(
            \boldsymbol{x} - \hat{\boldsymbol{x}}(\boldsymbol{\xi})
        \right) \rightarrow \min

    The stationary condition is the orthogonality of the distance vector and the
    tangent vectors
    :math:`\boldsymbol{a}_\alpha = \partial \hat{\boldsymbol{x}} / \partial \xi^\alpha`.
    The Hessian :math:`a_{\alpha\beta} - \boldsymbol{d} \cdot \hat{\boldsymbol{x}},
    _{\alpha\beta}` of the objective function is not positive definite if a point is
    located far away from a face. In this case, the Gauss-Newton approximation
    :math:`a_{\alpha\beta}` is used instead, which is always positive definite. This
    ensures a descent direction and hence a robust iteration for all pairs.
    """

    # the mixed second derivative of the surface coordinates is constant
    dadr = D2HDRDS @ vertices

    coordinates = np.zeros((len(points), 2))
    dcoordinates = np.zeros((len(points), 2))

    for _ in range(maxiter):
        h, dhdr = shape_function_quad(coordinates)

        # batched matrix products are used instead of einsum for performance reasons
        x = (h[:, None, :] @ vertices)[:, 0]
        a = dhdr.transpose(0, 2, 1) @ vertices
        d = points - x

        # negative gradient of the objective function
        fun = (a @ d[:, :, None])[:, :, 0]

        # hessian of the objective function and its Gauss-Newton approximation
        metric = a @ a.transpose(0, 2, 1)
        hessian = metric.copy()
        ddadr = np.sum(d * dadr, axis=-1)
        hessian[:, 0, 1] -= ddadr
        hessian[:, 1, 0] -= ddadr

        # use the Gauss-Newton approximation if the hessian is not positive definite
        determinant = (
            hessian[:, 0, 0] * hessian[:, 1, 1] - hessian[:, 0, 1] * hessian[:, 1, 0]
        )
        definite = (determinant > 0) & (hessian[:, 0, 0] > 0)
        hessian = np.where(definite[:, None, None], hessian, metric)

        dcoordinates = solve_2x2(hessian, fun)

        # limit the step size (trust region) and the range of the natural element
        # coordinates: only projections inside (or close to) a face are used
        np.clip(dcoordinates, -1.0, 1.0, out=dcoordinates)
        coordinates += dcoordinates
        np.clip(coordinates, -2.0, 2.0, out=coordinates)

        if np.all(np.abs(dcoordinates) < tol):
            break

    return coordinates, np.all(np.abs(dcoordinates) < np.sqrt(tol), axis=1)


class ContactSurfacePair:
    """A pair of a secondary (slave) and a primary (master) contact surface with
    methods to evaluate the contact kinematics and to assemble the sparse contact
    vector and matrix contributions.

    Parameters
    ----------
    field : FieldContainer
        A field container with a displacement field, created on a boundary region. The
        weak form of the contact is integrated on the faces of this secondary surface.
    field_primary : FieldContainer
        A field container with a displacement field, created on a boundary region. The
        integration points of the secondary surface are projected on the faces of this
        primary surface.
    weight : float, optional
        A scale factor for the contributions of this surface pair (default is 1.0).

    Notes
    -----
    This class is used internally by :class:`~felupe.SolidBodyContact`.
    """

    def __init__(self, field, field_primary, weight=1.0):
        self.field = field
        self.field_primary = field_primary
        self.weight = weight

        for f in [field, field_primary]:
            if not hasattr(f.region, "normals"):
                raise TypeError(
                    "A field on a boundary region is required, e.g. created on a "
                    "`RegionHexahedronBoundary`."
                )
            if f.region.mesh.cell_type != "hexahedron":
                raise NotImplementedError(
                    "Only boundary regions of hexahedron meshes are supported."
                )

        region = self.field.region
        region_primary = self.field_primary.region

        # the faces of a boundary region of a hexahedron mesh are bilinear quads. the
        # first four points of a boundary cell are the points of its face and the
        # values of the element shape functions are equal for all cells
        self.cells_faces = region.mesh.cells_faces
        self.cells_faces_primary = region_primary.mesh.cells_faces
        self.h = np.ascontiguousarray(region.h[:4, :, 0])

        # differential area of the secondary surface (reference configuration), the
        # weights of the quadrature scheme are already included
        self.dV = region.dV
        self.ncells = self.dV.shape[1]

        # the orientation of the vertices of the faces of a boundary region is not
        # necessarily aligned with the outward unit normal vectors of the region
        self.orientation = self._init_orientation(region_primary)

        # characteristic size of the faces of both surfaces
        self.size = min(
            np.sqrt(region.dV.sum(axis=0).mean()),
            np.sqrt(region_primary.dV.sum(axis=0).mean()),
        )

        self.points = np.unique(self.cells_faces)
        self.points_primary = np.unique(self.cells_faces_primary)

        self.area = region.dV.sum()
        self.area_primary = region_primary.dV.sum()

        self.dof = self.field[0].indices.dof
        self.ndof = self.field[0].indices.shape[0]

        # the face of the primary surface of the previous evaluation, which is used to
        # stabilize the face-assignment of the integration points
        self.face = np.full(self.dV.size, -1)

    def _init_orientation(self, region):
        "Return the orientation of the faces w.r.t. the outward unit normal vectors."

        vertices = region.mesh.points[region.mesh.cells_faces]
        normal = np.cross(
            vertices[:, 1] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]
        )

        # unit normal vectors of the boundary region, evaluated at the first
        # quadrature point of each face
        normals = region.normals[:, 0, :].T

        return np.sign(np.einsum("mi,mi->m", normal, normals))

    def kinematics(self, x, max_distance, candidates, tolerance, workers=1):
        r"""Return the contact kinematics, evaluated at the integration points of the
        faces of the secondary surface.

        Parameters
        ----------
        x : ndarray of shape (npoints, 3)
            The deformed coordinates of all points of the mesh.
        max_distance : float
            The maximum distance between an integration point and a face of the primary
            surface which is considered by the contact search.
        candidates : int
            The number of candidate faces of the primary surface per integration point
            of the secondary surface.
        tolerance : float
            The tolerance for the natural element coordinates of the projected points.
            A projection is only valid if the coordinates are within
            ``[-1 - tolerance, 1 + tolerance]``.
        workers : int, optional
            The number of workers used for the tree-query (default is 1).

        Returns
        -------
        dict or None
            A dict with the contact kinematics of the active integration points. None
            is returned if no integration point is in contact.
        """

        # deformed coordinates of the integration points of the secondary surface
        points = np.einsum("aq,cai->qci", self.h, x[self.cells_faces]).reshape(-1, 3)

        # deformed coordinates of the vertices of the faces of the primary surface
        vertices = x[self.cells_faces_primary]

        # broad-phase contact search: find the nearest faces of the primary surface by
        # a tree-query on the face centers
        center = vertices.mean(axis=1)
        radius = np.linalg.norm(vertices - center[:, None], axis=2).max(axis=1)

        tree = cKDTree(center)
        k = min(candidates, len(center))
        distance, face = tree.query(points, k=k, workers=workers)

        distance = distance.reshape(len(points), k)
        face = face.reshape(len(points), k)

        # discard pairs which are too far away
        mask = distance <= radius[face] + max_distance
        point = np.broadcast_to(np.arange(len(points))[:, None], (len(points), k))

        point = point[mask]
        face = face[mask]

        if len(point) == 0:
            return None

        # narrow-phase contact search: closest-point projection
        vertices_face = vertices[face]
        coordinates, converged = closest_point_projection(points[point], vertices_face)

        h, dhdr = shape_function_quad(coordinates)

        xp = (h[:, None, :] @ vertices_face)[:, 0]
        a = dhdr.transpose(0, 2, 1) @ vertices_face
        dadr = D2HDRDS @ vertices_face

        normal = np.cross(a[:, 0], a[:, 1])
        normal *= self.orientation[face][:, None]
        normal /= np.linalg.norm(normal, axis=1)[:, None]

        d = points[point] - xp
        gap = np.sum(d * normal, axis=-1)

        # the metric and the curvature of the primary surface. the modified metric
        # H = a - g * kappa is positive definite as long as the penetration is smaller
        # than the radius of curvature of the primary surface. only in this case, the
        # projection is a (local) minimum of the distance
        metric = a @ a.transpose(0, 2, 1)
        curvature = np.zeros((len(point), 2, 2))
        curvature[:, 0, 1] = curvature[:, 1, 0] = np.sum(normal * dadr, axis=-1)

        H = metric - gap[:, None, None] * curvature
        minimum = (H[:, 0, 0] * H[:, 1, 1] - H[:, 0, 1] * H[:, 1, 0] > 0) & (
            H[:, 0, 0] > 0
        )

        # faces which share at least one point are neighbours and must not be in
        # contact. this also removes the projections of a face on itself, which occur
        # if the masks of both boundary regions are overlapping
        cells = self.cells_faces[point % self.ncells]
        neighbour = np.any(
            cells[:, :, None] == self.cells_faces_primary[face][:, None, :],
            axis=(1, 2),
        )

        # the face of the previous evaluation is released with a doubled tolerance.
        # this hysteresis prevents an oscillating activation of integration points
        # which are projected near the boundary of the primary surface
        previous = self.face[point] == face
        released = np.where(previous, 2 * tolerance, tolerance)

        # a projection is only valid if it is located inside a face of the primary
        # surface and if the (signed) distance is within the search distance
        inside = np.all(np.abs(coordinates) <= 1 + released[:, None], axis=1)
        valid = (
            converged
            & minimum
            & inside
            & ~neighbour
            & (gap < 0)
            & (gap > -max_distance)
        )

        if not np.any(valid):
            return None

        # if an integration point is projected on more than one face, then the closest
        # face is used. the distance is measured to the closest point which is located
        # inside a face, i.e. with clipped natural element coordinates. this is
        # essential: a criterion which is based on the gap is ambiguous for integration
        # points which are located near the edges of the faces of the primary surface
        # and leads to an oscillating face-assignment between the iterations
        hc = shape_function_quad(np.clip(coordinates, -1.0, 1.0))[0]
        distance = np.linalg.norm(
            points[point] - (hc[:, None, :] @ vertices_face)[:, 0], axis=1
        )

        score = np.where(valid, distance, np.inf)

        # the face of the previous evaluation is kept as long as the projection is
        # still located inside this face. this hysteresis is essential for the
        # convergence: without it, the face-assignment of an integration point, which
        # is located on an edge or on a point of the primary surface, oscillates
        # between the iterations of the Newton-Raphson method
        score[valid & previous] = -1.0

        order = np.lexsort((score, point))

        best = np.ones(len(order), dtype=bool)
        best[1:] = point[order][1:] != point[order][:-1]
        best = order[best]
        best = best[valid[best]]

        point, face = point[best], face[best]
        gap, normal, a = gap[best], normal[best], a[best]
        metric, curvature = metric[best], curvature[best]
        h, dhdr = h[best], dhdr[best]

        # store the face-assignment for the next evaluation
        self.face[:] = -1
        self.face[point] = face

        return {
            "point": point,
            "face": face,
            "gap": gap,
            "normal": normal,
            "tangents": a,
            "metric": metric,
            "curvature": curvature,
            "h": h,
            "dhdr": dhdr,
        }

    def variations(self, kinematics):
        r"""Return the variations of the gap and of the surface quantities of the
        primary surface w.r.t. the displacements of the points of a face-pair.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see :meth:`~ContactSurfacePair.kinematics`.

        Returns
        -------
        b : list of ndarray of shape (npairs, 12)
            The variation of the gap :math:`\delta g = \boldsymbol{b} \cdot \delta
            \boldsymbol{u}` for the secondary and the primary surface.
        A : ndarray of shape (npairs, 2, 12)
            The variation :math:`A_\alpha = \boldsymbol{n} \cdot \delta
            \boldsymbol{a}_\alpha` of the primary surface.
        B : list of ndarray of shape (npairs, 2, 12)
            The variation :math:`B_\alpha = \boldsymbol{a}_\alpha \cdot \delta
            \boldsymbol{d}` for the secondary and the primary surface.

        Notes
        -----
        The variation of the gap does not include the variation of the unit normal
        vector nor the variation of the projected coordinates, because both are
        orthogonal to the gap vector :math:`\boldsymbol{d} = g\ \boldsymbol{n}`.

        ..  math::

            \delta g = \left(
                \delta \boldsymbol{u} - \delta \bar{\boldsymbol{u}}
            \right) \cdot \boldsymbol{n}
        """

        point = kinematics["point"]
        normal = kinematics["normal"]
        tangents = kinematics["tangents"]

        q, c = np.divmod(point, self.ncells)

        # shape functions of the secondary (h) and the primary (hp) faces
        h = self.h[:, q].T
        hp = kinematics["h"]
        dhdr = kinematics["dhdr"]

        npairs = len(point)

        b = [
            (h[:, :, None] * normal[:, None, :]).reshape(npairs, 12),
            (-hp[:, :, None] * normal[:, None, :]).reshape(npairs, 12),
        ]
        B = [
            (h[:, None, :, None] * tangents[:, :, None, :]).reshape(npairs, 2, 12),
            (-hp[:, None, :, None] * tangents[:, :, None, :]).reshape(npairs, 2, 12),
        ]
        A = (dhdr.transpose(0, 2, 1)[:, :, :, None] * normal[:, None, None, :]).reshape(
            npairs, 2, 12
        )

        return b, A, B

    def assemble_vector(self, kinematics, gradient, parallel=False):
        r"""Return the assembled sparse contact force vector.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see :meth:`~ContactSurfacePair.kinematics`.
        gradient : ndarray of shape (npairs,)
            The first derivative :math:`\Phi'(g)` of the contact potential w.r.t. the
            gap, i.e. the negative contact pressure.
        parallel : bool, optional
            Flag to activate a threaded assembly (default is False).

        Returns
        -------
        scipy.sparse.csr_matrix
            The assembled sparse contact force vector.

        Notes
        -----
        The contribution of the secondary surface is assembled by a weak form, see Eq.
        :eq:`contact-weak-form`, where the contact traction is integrated on the faces
        of the secondary surface. The equal and opposite contribution of the primary
        surface is evaluated at the projected points and is assembled directly.

        ..  math::
            :label: contact-weak-form

            \delta \Pi_c = \int_{\Gamma} \Phi'(g) \left(
                \delta \boldsymbol{u} - \delta \bar{\boldsymbol{u}}
            \right) \cdot \boldsymbol{n}\ d\Gamma
        """

        point = kinematics["point"]
        normal = kinematics["normal"]
        q, c = np.divmod(point, self.ncells)

        # secondary surface: the traction at the integration points of the faces is
        # integrated by a weak form
        traction = np.zeros((3, *self.dV.shape))
        traction[:, q, c] = self.weight * gradient * normal.T

        force = IntegralForm(
            fun=[traction], v=self.field, dV=self.dV, grad_v=[False]
        ).assemble(parallel=parallel)

        # primary surface: the traction is evaluated at the projected points
        dA = self.weight * self.dV[q, c] * gradient
        values = -dA[:, None, None] * kinematics["h"][:, :, None] * normal[:, None, :]
        rows = self.dof[self.cells_faces_primary[kinematics["face"]]]

        force += csr_matrix(
            (values.ravel(), (rows.ravel(), np.zeros(rows.size, dtype=int))),
            shape=(self.ndof, 1),
        )

        return force

    def assemble_matrix(
        self, kinematics, gradient, hessian, parallel=False, geometric=True
    ):
        r"""Return the assembled sparse contact stiffness matrix.

        Parameters
        ----------
        kinematics : dict
            The contact kinematics, see :meth:`~ContactSurfacePair.kinematics`.
        gradient : ndarray of shape (npairs,)
            The first derivative :math:`\Phi'(g)` of the contact potential w.r.t. the
            gap, i.e. the negative contact pressure.
        hessian : ndarray of shape (npairs,)
            The second derivative :math:`\Phi''(g)` of the contact potential w.r.t. the
            gap.
        parallel : bool, optional
            Flag to activate a threaded assembly (default is False).
        geometric : bool, optional
            Flag to add the geometric part of the contact stiffness matrix (default is
            True).

        Returns
        -------
        scipy.sparse.csr_matrix
            The assembled sparse contact stiffness matrix.

        Notes
        -----
        The linearization of the variation of the gap is given in Eq.
        :eq:`contact-linearization` with the modified metric
        :math:`H_{\alpha\beta} = a_{\alpha\beta} - g\ \kappa_{\alpha\beta}` and
        :math:`M^{\alpha\beta} = a^{\alpha\gamma} \kappa_{\gamma\delta} H^{\delta\beta}`
        . The resulting stiffness matrix is symmetric.

        ..  math::
            :label: contact-linearization

            \Delta \delta g = &-H^{\alpha\beta} \left(
                A_\alpha \Delta B_\beta + B_\alpha \Delta A_\beta
            \right)

            &- g\ H^{\alpha\beta} A_\alpha \Delta A_\beta
             - M^{\alpha\beta} B_\alpha \Delta B_\beta
        """

        point = kinematics["point"]
        gap = kinematics["gap"]
        normal = kinematics["normal"]
        tangents = kinematics["tangents"]
        metric = kinematics["metric"]
        curvature = kinematics["curvature"]

        q, c = np.divmod(point, self.ncells)
        dA = self.weight * self.dV[q, c]

        # inverse of the modified metric and the curvature-related fourth-order term
        inverse_metric = invert_2x2(metric)
        inverse_H = invert_2x2(metric - gap[:, None, None] * curvature)
        M = np.einsum("pJK,pKL,pLM->pJM", inverse_metric, curvature, inverse_H)

        # secondary surface: the (symmetric) block of the stiffness matrix, which
        # contains only test- and trial-functions of the secondary surface, is
        # assembled by a weak form
        elasticity = np.zeros((3, 3, *self.dV.shape))
        elasticity[:, :, q, c] = self.weight * (
            hessian * np.einsum("pi,pj->ijp", normal, normal)
        )
        if geometric:
            elasticity[:, :, q, c] -= self.weight * (
                gradient * np.einsum("pJK,pJi,pKj->ijp", M, tangents, tangents)
            )

        stiffness = IntegralForm(
            fun=[elasticity],
            v=self.field,
            u=self.field,
            dV=self.dV,
            grad_v=[False],
            grad_u=[False],
        ).assemble(parallel=parallel)

        # coupling- and primary-blocks of the stiffness matrix
        b, A, B = self.variations(kinematics)

        # material part
        Ksm = hessian[:, None, None] * b[0][:, :, None] * b[1][:, None, :]
        Kmm = hessian[:, None, None] * b[1][:, :, None] * b[1][:, None, :]

        if geometric:
            HA = np.einsum("pJK,pKi->pJi", inverse_H, A)
            MB = [np.einsum("pJK,pKi->pJi", M, Bi) for Bi in B]

            # geometric part of the secondary-primary coupling block
            Ksm -= gradient[:, None, None] * (
                np.einsum("pJi,pJj->pij", B[0], HA)
                + np.einsum("pJi,pJj->pij", MB[0], B[1])
            )

            # geometric part of the primary-primary block
            T = np.einsum("pJi,pJj->pij", HA, B[1])
            Kmm -= gradient[:, None, None] * (
                T
                + T.transpose(0, 2, 1)
                + gap[:, None, None] * np.einsum("pJi,pJj->pij", HA, A)
                + np.einsum("pJi,pJj->pij", MB[1], B[1])
            )

        Ksm *= dA[:, None, None]
        Kmm *= dA[:, None, None]

        rows = self.dof[self.cells_faces[c]].reshape(-1, 12)
        cols = self.dof[self.cells_faces_primary[kinematics["face"]]].reshape(-1, 12)

        Ksm = self._assemble(Ksm, rows, cols)
        stiffness += Ksm + Ksm.T + self._assemble(Kmm, cols, cols)

        return stiffness

    def _assemble(self, values, rows, cols):
        "Return a sparse matrix, assembled from dense sub-matrices of face-pairs."

        return csr_matrix(
            (
                values.ravel(),
                (
                    np.broadcast_to(rows[:, :, None], values.shape).ravel(),
                    np.broadcast_to(cols[:, None, :], values.shape).ravel(),
                ),
            ),
            shape=(self.ndof, self.ndof),
        )


class SolidBodyContact:
    r"""A frictionless three-dimensional contact between the surfaces of two solid
    bodies.

    Parameters
    ----------
    field : FieldContainer
        A field container with a displacement field, created on a boundary region of
        the secondary (slave) surface, e.g. on a
        :class:`~felupe.RegionHexahedronBoundary`. The weak form of the contact is
        integrated on the faces of this surface.
    field_primary : FieldContainer
        A field container with a displacement field, created on a boundary region of
        the primary (master) surface. The integration points of the secondary surface
        are projected on the faces of this surface.
    items : list of SolidBody or None, optional
        A list of items which are used to estimate the penalty stiffness (default is
        None). If None, ``penalty`` must be given.
    penalty : float or None, optional
        The penalty stiffness :math:`\epsilon` as contact traction per unit penetration
        (default is None). If None, the penalty stiffness is estimated from the mean
        stiffness of the degrees of freedom on both contact surfaces, see Eq.
        :eq:`contact-penalty`.
    penalty_scale : float, optional
        A scale factor which is applied on the estimated penalty stiffness (default is
        10.0). This has no effect if ``penalty`` is given. Increase this factor to
        reduce the penetration, decrease it if the Newton-Raphson method does not
        converge.
    smoothing : float or None, optional
        The length :math:`\delta` of the transition zone of the regularized penalty
        law, see Eq. :eq:`contact-pressure` (default is None). If None, a fraction of
        the characteristic size of the faces of the contact surfaces is used. A value
        of zero deactivates the regularization.
    two_pass : bool, optional
        Flag to evaluate the contact twice with exchanged roles of the surfaces, each
        with half of the contact potential (default is False). This removes the bias
        which is introduced by the choice of the secondary surface at the expense of a
        doubled evaluation time.
    max_distance : float or None, optional
        The maximum distance between an integration point of the secondary surface and
        a face of the primary surface which is considered by the contact search
        (default is None). If None, five times the characteristic size of the faces of
        the contact surfaces is used. This limits the maximum detectable penetration.
    candidates : int, optional
        The number of candidate faces of the primary surface which are evaluated per
        integration point of the secondary surface (default is 8).
    tolerance : float, optional
        The relative tolerance for the natural element coordinates of the projected
        points (default is 0.1). A projection is valid if its coordinates are within
        ``[-1 - tolerance, 1 + tolerance]``.
    geometric_stiffness : bool, optional
        Flag to add the geometric part of the contact stiffness matrix (default is
        True). This is required for a quadratic rate of convergence.

    Attributes
    ----------
    penalty : float or None
        The penalty stiffness. This is None until it is estimated on the first assembly.
    results : Results
        The results of the contact, e.g. the gap and the contact pressure at the
        integration points of the secondary surface.

    Notes
    -----
    The contact constraints are enforced by a penalty regularization of the contact
    potential :math:`\Pi_c`, which is integrated on the faces of the secondary surface,
    see Eq. :eq:`contact-potential`. This is a segment-to-segment (surface-to-surface)
    formulation: the contact tractions are evaluated at the integration points of the
    secondary surface and are not lumped to its points.

    ..  math::
        :label: contact-potential

        \Pi_c = \int_\Gamma \Phi(g)\ d\Gamma

    The gap :math:`g` is evaluated by a closest-point projection of the deformed
    coordinates :math:`\boldsymbol{x}` of the integration points of the secondary
    surface onto the deformed primary surface, see Eq. :eq:`contact-gap`. The projected
    coordinates :math:`\bar{\boldsymbol{x}}` and the outward unit normal vector
    :math:`\boldsymbol{n}` of the primary surface are evaluated at the natural element
    coordinates of the projection.

    ..  math::
        :label: contact-gap

        g = \left( \boldsymbol{x} - \bar{\boldsymbol{x}} \right) \cdot \boldsymbol{n}

    The contact pressure :math:`p = -\Phi'(g)` is a regularized penalty law with a
    smooth transition of length :math:`\delta`, see Eq. :eq:`contact-pressure`. In
    contrast to the non-regularized penalty law, both the contact pressure and its
    derivative w.r.t. the gap are continuous. This removes the jump of the tangent
    stiffness at the activation of a contact and hence improves the rate of convergence
    of the Newton-Raphson method significantly.

    ..  math::
        :label: contact-pressure

        p(g) = \begin{cases}
            -\epsilon \left( g + \dfrac{\delta}{2} \right) & g \le -\delta \\
            \epsilon\ \dfrac{g^2}{2 \delta} & -\delta < g < 0 \\
            0 & g \ge 0
        \end{cases}

    If no penalty stiffness is given, it is estimated from the mean of the diagonal
    entries :math:`\bar{k}` of the stiffness matrices of the given ``items``, evaluated
    on the degrees of freedom of the points of a contact surface, and the mean area
    :math:`\bar{a} = A / n_{points}` per point of this surface, see Eq.
    :eq:`contact-penalty`. The softer of both contact surfaces is decisive. This
    estimate scales with :math:`E / h` and hence requires no manual tuning, neither for
    rubber-to-rubber nor for rubber-to-metal contacts.

    ..  math::
        :label: contact-penalty

        \epsilon = \text{scale} \cdot \min{\left(
            \frac{\bar{k}}{\bar{a}},\ \frac{\bar{k}_{primary}}{\bar{a}_{primary}}
        \right)}

    ..  note::

        The mesh of both boundary regions must be the same mesh as the mesh of the
        region of the solid bodies. Two separate meshes are combined by
        :meth:`MeshContainer.stack() <felupe.MeshContainer.stack>`. Both contact
        surfaces must be disjoint, i.e. they must not share any points. Faces which
        share at least one point are treated as neighbours and are never in contact.

    ..  hint::

        The choice of the secondary surface matters for a single-pass contact: Use the
        softer and finer meshed surface as the secondary surface, e.g. the rubber
        surface of a rubber-to-metal contact. Alternatively, use ``two_pass=True``,
        which removes this bias but requires about twice the evaluation time.

    Examples
    --------
    A rubber block is pressed on a (stiffer) block. Both blocks are meshed
    individually and are combined to a single mesh.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> bottom = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=(4, 4, 3))
        >>> top = fem.Cube(a=(0.15, 0.15, 1.02), b=(0.85, 0.85, 1.62), n=(3, 3, 3))
        >>> container = fem.MeshContainer([bottom, top], merge=True)
        >>> mesh = container.stack()
        >>>
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=50.0), field=field)

    The contact surfaces are created as boundary regions on the same mesh. Only the
    faces on the outline of the mesh which are located in the region of interest are
    used.

    ..  pyvista-plot::
        :context:

        >>> mask = np.logical_and(mesh.z > 1.0, mesh.z < 1.1)
        >>> secondary = fem.FieldContainer(
        ...     [fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mask), dim=3)]
        ... )
        >>> primary = fem.FieldContainer(
        ...     [fem.Field(
        ...         fem.RegionHexahedronBoundary(mesh, mask=np.isclose(mesh.z, 1.0)),
        ...         dim=3,
        ...     )]
        ... )
        >>> contact = fem.SolidBodyContact(secondary, primary, items=[solid])

    The bottom face of the lower block is fixed and the top face of the upper block is
    moved downwards.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {
        ...     "fixed": fem.Boundary(field[0], fz=0.0),
        ...     "clamped": fem.Boundary(field[0], fz=mesh.z.max(), skip=(0, 0, 1)),
        ...     "move": fem.Boundary(field[0], fz=mesh.z.max(), skip=(1, 1, 0)),
        ... }
        >>> move = fem.math.linsteps([0, -0.3], num=3)
        >>> step = fem.Step(
        ...     items=[solid, contact],
        ...     ramp={boundaries["move"]: move},
        ...     boundaries=boundaries,
        ... )
        >>> job = fem.Job(steps=[step]).evaluate(verbose=0)

    The number of integration points which are in contact and the penalty stiffness,
    which is estimated from the given items, are available in the contact object.

    ..  pyvista-plot::
        :context:

        >>> contact.results.npoints_in_contact
        16

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> solid.plot("Principal Values of Cauchy Stress").show()

    See Also
    --------
    felupe.ContactRigidPlane : A node-to-surface contact, where the surface is given by
        a rigid plane.
    """

    def __init__(
        self,
        field,
        field_primary,
        items=None,
        penalty=None,
        penalty_scale=10.0,
        smoothing=None,
        two_pass=False,
        max_distance=None,
        candidates=8,
        tolerance=0.1,
        geometric_stiffness=True,
    ):
        self.field = field
        self.field_primary = field_primary
        self.items = items

        self.penalty = penalty
        self.penalty_scale = penalty_scale
        self.two_pass = two_pass
        self.candidates = candidates
        self.tolerance = tolerance
        self.geometric_stiffness = geometric_stiffness

        if items is None and penalty is None:
            raise ValueError("Either `items` or `penalty` must be given.")

        self.pairs = [ContactSurfacePair(field, field_primary)]

        if two_pass:
            self.pairs = [
                ContactSurfacePair(field, field_primary, weight=0.5),
                ContactSurfacePair(field_primary, field, weight=0.5),
            ]

        self.size = self.pairs[0].size

        self.smoothing = smoothing
        if smoothing is None:
            self.smoothing = 1e-2 * self.size

        self.max_distance = max_distance
        if max_distance is None:
            self.max_distance = 5 * self.size

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.results = Results()
        self.results.gap = []
        self.results.pressure = []
        self.results.npoints_in_contact = 0

        self._kinematics = None
        self._values = None

    def __repr__(self):
        header = "<felupe SolidBodyContact object>"
        penalty = f"  Penalty stiffness: {self.penalty}"
        smoothing = f"  Smoothing: {self.smoothing}"
        contact = f"  Integration points in contact: {self.results.npoints_in_contact}"

        return "\n".join([header, penalty, smoothing, contact])

    def pressure(self, gap):
        r"""Return the contact pressure and its derivative w.r.t. the gap, evaluated by
        a regularized penalty law, see Eq. :eq:`contact-pressure`.

        Parameters
        ----------
        gap : ndarray
            The gap between the contact surfaces.

        Returns
        -------
        pressure : ndarray
            The contact pressure.
        dpressure : ndarray
            The derivative of the contact pressure w.r.t. the gap.
        """

        penalty, smoothing = self.penalty, self.smoothing

        pressure = np.zeros_like(gap)
        dpressure = np.zeros_like(gap)

        if smoothing > 0:
            closed = gap <= -smoothing
            transition = np.logical_and(gap > -smoothing, gap < 0)

            g = gap[transition]
            pressure[transition] = penalty * g**2 / (2 * smoothing)
            dpressure[transition] = penalty * g / smoothing

            pressure[closed] = -penalty * (gap[closed] + smoothing / 2)
        else:
            closed = gap < 0
            pressure[closed] = -penalty * gap[closed]

        dpressure[closed] = -penalty

        return pressure, dpressure

    def update_penalty(self):
        r"""Estimate and update the penalty stiffness from the mean stiffness of the
        degrees of freedom on the contact surfaces, see Eq. :eq:`contact-penalty`.

        Returns
        -------
        float
            The estimated penalty stiffness.
        """

        pair = self.pairs[0]
        diagonal = np.zeros(pair.ndof)

        for item in self.items:
            stiffness = item.results.stiffness

            if stiffness is None:
                stiffness = item.assemble.matrix()

            if item.assemble.multiplier is not None:
                stiffness = stiffness * item.assemble.multiplier

            values = stiffness.diagonal()
            size = min(len(values), pair.ndof)
            diagonal[:size] += values[:size]

        surfaces = [
            (pair.points, pair.area),
            (pair.points_primary, pair.area_primary),
        ]

        penalty = []
        for points, area in surfaces:
            stiffness = np.abs(diagonal[pair.dof[points].ravel()]).mean()
            penalty.append(stiffness * len(points) / area)

        self.penalty = self.penalty_scale * min(penalty)

        return self.penalty

    def _extract(self, field=None, parallel=False):
        "Evaluate and cache the contact kinematics of all surface pairs."

        if field is not None:
            self.field = field

        values = self.field[0].values
        self.field_primary[0].values = values

        if self._kinematics is not None and np.array_equal(self._values, values):
            return self._kinematics

        if self.penalty is None:
            self.update_penalty()

        x = self.field.region.mesh.points + values

        self._kinematics = [
            pair.kinematics(
                x=x,
                max_distance=self.max_distance,
                candidates=self.candidates,
                tolerance=self.tolerance,
                workers=-1 if parallel else 1,
            )
            for pair in self.pairs
        ]
        self._values = values.copy()

        self.results.gap = [
            None if kin is None else kin["gap"] for kin in self._kinematics
        ]
        self.results.pressure = [
            None if kin is None else self.pressure(kin["gap"])[0]
            for kin in self._kinematics
        ]
        self.results.npoints_in_contact = sum(
            [0 if kin is None else len(kin["gap"]) for kin in self._kinematics]
        )

        return self._kinematics

    def _vector(self, field=None, parallel=False, resize=None):
        "Assemble the sparse contact force vector."

        kinematics = self._extract(field=field, parallel=parallel)
        force = csr_matrix((self.pairs[0].ndof, 1))

        for pair, kin in zip(self.pairs, kinematics):
            if kin is None:
                continue

            pressure = self.pressure(kin["gap"])[0]
            force += pair.assemble_vector(kin, -pressure, parallel=parallel)

        if resize is not None:
            force.resize(*resize.shape)

        self.results.force = force

        return force

    def _matrix(self, field=None, parallel=False, resize=None):
        "Assemble the sparse contact stiffness matrix."

        kinematics = self._extract(field=field, parallel=parallel)
        ndof = self.pairs[0].ndof
        stiffness = csr_matrix((ndof, ndof))

        for pair, kin in zip(self.pairs, kinematics):
            if kin is None:
                continue

            pressure, dpressure = self.pressure(kin["gap"])
            stiffness += pair.assemble_matrix(
                kin,
                -pressure,
                -dpressure,
                parallel=parallel,
                geometric=self.geometric_stiffness,
            )

        if resize is not None:
            stiffness.resize(*resize.shape)

        self.results.stiffness = stiffness

        return stiffness

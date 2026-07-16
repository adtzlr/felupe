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
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

from ._helpers import Assemble, Results


class SolidBodyContact:
    r"""A frictional node-to-segment penalty contact between two deformable solid
    bodies. Both surfaces are given by field containers, whose fields are defined on a
    :class:`~felupe.RegionBoundary` (e.g. :class:`~felupe.RegionQuadBoundary` or
    :class:`~felupe.RegionHexahedronBoundary`).

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field, defined on a
        boundary region. The points of this surface are treated as slave (contactor)
        points.
    other_field : FieldContainer
        A field container with the displacement field as first field, defined on a
        boundary region. The facets of this surface are treated as master (target)
        segments.
    friction : float, optional
        Coulomb friction coefficient :math:`\mu`. Default is 0.0.
    multiplier : float, optional
        A scale factor for the normal penalty stiffness. Default is 1.0. If ``items``
        are given, the final penalty is scaled by the mean absolute diagonal stiffness
        of the items restricted to the contact degrees of freedom.
    multiplier_tangential : float, optional
        A scale factor for the tangential penalty stiffness, relative to the normal
        penalty stiffness. Default is 0.1.
    items : list or None, optional
        A list of items (e.g. :class:`~felupe.SolidBody`) which are used to evaluate a
        mean stiffness-based penalty. If None, the penalty is not based on the mean
        stiffness. Default is None.
    symmetric : bool, optional
        A flag to enable a symmetric two-pass contact search, where the roles of slave
        and master surface are additionally swapped in a second pass. Default is False.
    capture : float, optional
        The search (capture) distance for the contact detection, in units of the mean
        master-segment size. A slave-point is only considered to be in contact with a
        segment if its projection is closer than this distance. This avoids spurious
        contact of far-away points which happen to lie on the inner side of the infinite
        plane of a distant segment. Default is 2.0.

    Notes
    -----
    A :class:`~felupe.SolidBodyContact` is supported as an item in a
    :class:`~felupe.Step`. It provides the assemble-methods
    :meth:`SolidBodyContact.assemble.vector() <felupe.SolidBodyContact.assemble.vector>`
    and
    :meth:`SolidBodyContact.assemble.matrix() <felupe.SolidBodyContact.assemble.matrix>`.

    ..  note::

        Both bodies must share a common (merged) point numbering, i.e. their meshes
        should be part of one :class:`~felupe.MeshContainer` created with ``merge=True``
        and a top-level field on the stacked mesh has to be passed to the job, e.g.
        ``job.evaluate(x0=field)``.

    ..  note::

        The contact formulation is based on a penalty method. The penalty stiffness
        (scaled by ``multiplier``) should be chosen sufficiently large to limit the
        penetration but not too large to avoid ill-conditioning and active-set
        chattering. Best practice is to provide ``items`` (the solid bodies which are
        connected to the contact surfaces), which enables a mean stiffness-based
        penalty. On finer meshes, smaller load-increments improve the robustness of the
        Newton-Raphson iterations.

    The contact formulation is based on a penalty method. For each slave point the
    closest master segment is found by a closest-point-projection in the deformed
    configuration. The gap is evaluated in the direction of the (outward) master surface
    normal. If the gap is negative, the contact is active, see Eq. :eq:`sbc-gap`.

    ..  math::
        :label: sbc-gap

        \boldsymbol{x}_m &= \sum_a N_a(\xi)\ \boldsymbol{x}_a

        g &= (\boldsymbol{x}_s - \boldsymbol{x}_m) \cdot \boldsymbol{n}

        g &\lt 0 \quad \text{(contact active)}

    The contact normal force is evaluated as a penalty contribution proportional to the
    gap and distributed to the slave point and the nodes of the master segment by the
    segment shape functions :math:`N_a(\xi)`, see Eq. :eq:`sbc-force`.

    ..  math::
        :label: sbc-force

        \boldsymbol{f}_s &= \varepsilon_n\ g\ \boldsymbol{n}

        \boldsymbol{f}_a &= -N_a(\xi)\ \varepsilon_n\ g\ \boldsymbol{n}

    The tangential contact friction forces are evaluated according to a Coulomb friction
    law, analogous to :class:`~felupe.ContactRigidPlane`. The tangent stiffness matrix
    is based on the dominant penalty contributions (evaluated for a fixed normal vector
    and fixed segment shape functions). Curvature terms of the master segment are
    neglected.

    Examples
    --------
    This example shows a frictionless contact between two blocks with a small initial
    gap. Two meshes are merged into one :class:`~felupe.MeshContainer` to obtain a
    common point numbering. The initial gap ensures that the two contact surfaces are
    not merged (bonded).

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> bottom = fem.Rectangle(a=(0, 0), b=(1, 1), n=(6, 6))
        >>> top = fem.Rectangle(a=(0, 1.1), b=(1, 2.1), n=(6, 6))
        >>> container = fem.MeshContainer([bottom, top], merge=True)
        >>>
        >>> regions = [fem.RegionQuad(m) for m in container.meshes]
        >>> fields = [
        ...     fem.FieldContainer([fem.FieldPlaneStrain(r, dim=2)]) for r in regions
        ... ]
        >>>
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> solids = [fem.SolidBody(umat, f) for f in fields]

    The contact surfaces are the top edge of the lower block and the bottom edge of the
    upper block. For each surface a boundary region and a boundary field are created.

    ..  pyvista-plot::
        :context:

        >>> mask = container.meshes[0].points[:, 1] == 1
        >>> boundary_bottom = fem.RegionQuadBoundary(
        ...     container.meshes[0], mask=mask, ensure_3d=False
        ... )
        >>> field_bottom = fem.FieldContainer(
        ...     [fem.FieldPlaneStrain(boundary_bottom, dim=2)]
        ... )
        >>>
        >>> mask = container.meshes[1].points[:, 1] == 1.1
        >>> boundary_top = fem.RegionQuadBoundary(
        ...     container.meshes[1], mask=mask, ensure_3d=False
        ... )
        >>> field_top = fem.FieldContainer([fem.FieldPlaneStrain(boundary_top, dim=2)])
        >>>
        >>> contact = fem.SolidBodyContact(
        ...     field_bottom, field_top, items=solids, multiplier=5.0
        ... )

    A top-level field on the stacked mesh is created and used for the boundary
    conditions and Newton's method. The lower block is fixed and the upper block is
    moved downwards.

    ..  pyvista-plot::
        :context:

        >>> region = fem.RegionQuad(container.stack())
        >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        >>>
        >>> boundaries = {
        ...     "fixed": fem.Boundary(field[0], fy=0),
        ...     "move": fem.Boundary(field[0], fy=2.1),
        ... }
        >>> move = fem.math.linsteps([0, -0.2], num=10)
        >>> step = fem.Step(
        ...     items=[*solids, contact],
        ...     ramp={boundaries["move"]: move},
        ...     boundaries=boundaries,
        ... )
        >>> job = fem.Job(steps=[step]).evaluate(x0=field)

    See Also
    --------
    felupe.ContactRigidPlane : A node-to-surface contact with a rigid plane.
    felupe.MultiPointContact : A frictionless point-to-rigid (wall) contact.
    """

    def __init__(
        self,
        field,
        other_field,
        friction=0.0,
        multiplier=1.0,
        multiplier_tangential=0.1,
        items=None,
        symmetric=False,
        capture=2.0,
    ):
        self.field = field
        self.other_field = other_field
        self.mesh = field.region.mesh
        self.dim = self.mesh.dim

        self.friction = friction
        self.multiplier = multiplier
        self.multiplier_tangential = multiplier_tangential
        self.items = items
        self.symmetric = symmetric
        self.capture = capture

        # undeformed (global) point coordinates
        self.points = self.mesh.points

        # extract points, cells and undeformed normals of a boundary region
        slave = self._surface(field.region)
        master = self._surface(other_field.region)

        # capture (search) distance per segment-surface, based on the mean facet-size.
        # A slave-point is only in contact with a segment if the projection is closer
        # than this distance - this avoids spurious contact of far-away points, which
        # happen to lie on the inner side of the infinite plane of a segment.
        slave_capture = self.capture * self._facet_size(slave["cells"])
        master_capture = self.capture * self._facet_size(master["cells"])

        # a pass is defined by a set of slave-points and master-segments
        self._passes = [
            (slave["points"], master["cells"], master["normals"], master_capture)
        ]
        if self.symmetric:
            self._passes.append(
                (master["points"], slave["cells"], slave["normals"], slave_capture)
            )
        self._pass_cache = [
            {
                "nodes": np.asarray(nodes, dtype=int),
                "corners": np.asarray(corners, dtype=int),
                "normals_ref": np.asarray(normals_ref, dtype=float),
                "capture": float(capture),
            }
            for nodes, corners, normals_ref, capture in self._passes
        ]

        # per-pass history (reference gap-vectors, active- and slip-state)
        self._states = [
            {
                "dx_ref": np.zeros((len(points), self.dim)),
                "active": np.zeros(len(points), dtype=bool),
                "slip": np.zeros(len(points), dtype=bool),
            }
            for points, *_ in self._passes
        ]

        # expose the first pass for introspection / plotting
        self.results = Results(stress=False, elasticity=False)
        self.results.dx_ref = self._states[0]["dx_ref"]
        self.results.active = self._states[0]["active"]
        self.results.slip = self._states[0]["slip"]

        # cache of contact degrees of freedom for the penalty scaling
        contact_points = np.unique(np.concatenate([slave["points"], master["points"]]))
        self._contact_dof = field[0].indices.dof[contact_points].ravel()

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    @staticmethod
    def _surface(region):
        "Extract surface point-ids, cells and undeformed normals of a boundary region."

        return {
            "points": np.unique(region.mesh.cells_faces),
            "cells": region.mesh.cells_faces,
            "normals": region.normals[:, 0].T,
        }

    def _facet_size(self, cells):
        "Return the mean length of the first edge of the given facets."

        edges = self.points[cells[:, 1]] - self.points[cells[:, 0]]
        return np.linalg.norm(edges, axis=1).mean()

    def _penalties(self):
        "Return the normal and tangential penalty stiffness scale factors."

        base = 1.0

        if self.items is not None and len(self._contact_dof) > 0:
            values = []
            for item in self.items:
                stiffness = item.results.stiffness
                if stiffness is None:
                    stiffness = item.assemble.matrix()
                diagonal = np.abs(stiffness.diagonal())
                values.append(diagonal[self._contact_dof].mean())
            base = float(np.mean(values))

        eps_n = self.multiplier * base
        eps_t = self.multiplier_tangential * eps_n

        return eps_n, eps_t

    def _project(self, xs, corners_coords, normal_ref):
        """Closest-point-projection of a slave point onto a master facet.

        Returns a dict with the projected point, the outward unit normal, the segment
        shape-functions, the signed gap and a flag if the projection is inside the
        facet - or None if the facet is degenerated.
        """

        tol = 1e-4

        if self.dim == 2:
            a, b = corners_coords[0], corners_coords[1]
            tangent = b - a
            length_squared = tangent @ tangent

            if length_squared == 0.0:
                return None

            xi = (xs - a) @ tangent / length_squared
            inside = (-tol <= xi) and (xi <= 1.0 + tol)
            xi = min(max(xi, 0.0), 1.0)

            projection = a + xi * tangent
            normal = np.array([tangent[1], -tangent[0]])
            normal = normal / np.linalg.norm(normal)

            if normal @ normal_ref < 0:
                normal = -normal

            shape = np.array([1.0 - xi, xi])

        else:
            # local Newton iterations for the bilinear-quad projection
            xi = np.zeros(2)
            for _ in range(8):
                shape, dshape = self._bilinear(xi)
                projection = shape @ corners_coords
                t1 = dshape[0] @ corners_coords
                t2 = dshape[1] @ corners_coords
                residual = np.array([(xs - projection) @ t1, (xs - projection) @ t2])
                jacobian = -np.array([[t1 @ t1, t1 @ t2], [t2 @ t1, t2 @ t2]])
                if np.linalg.det(jacobian) == 0.0:
                    return None
                dxi = np.linalg.solve(jacobian, residual)
                xi = xi - dxi
                if np.linalg.norm(dxi) < 1e-10:
                    break

            inside = bool(np.all(np.abs(xi) <= 1.0 + tol))
            xi = np.clip(xi, -1.0, 1.0)
            shape, dshape = self._bilinear(xi)
            projection = shape @ corners_coords
            t1 = dshape[0] @ corners_coords
            t2 = dshape[1] @ corners_coords
            normal = np.cross(t1, t2)
            norm = np.linalg.norm(normal)

            if norm == 0.0:
                return None

            normal = normal / norm

            if normal @ normal_ref < 0:
                normal = -normal

        gap_vector = xs - projection
        gap = gap_vector @ normal

        return {
            "normal": normal,
            "shape": shape,
            "gap": gap,
            "gap_vector": gap_vector,
            "inside": inside,
        }

    @staticmethod
    def _bilinear(xi):
        "Bilinear quad shape-functions and their derivatives at natural coord xi."

        a, b = xi
        shape = 0.25 * np.array(
            [(1 - a) * (1 - b), (1 + a) * (1 - b), (1 + a) * (1 + b), (1 - a) * (1 + b)]
        )
        dshape = 0.25 * np.array(
            [
                [-(1 - b), (1 - b), (1 + b), -(1 + b)],
                [-(1 - a), -(1 + a), (1 + a), (1 - a)],
            ]
        )
        return shape, dshape

    def _closest(self, xs, corners_all, normals_ref, displacement, capture):
        """Find the closest master facet with an interior projection for a slave point.

        Only facets on which the slave-point projects onto the interior of the segment
        (within a small tolerance) and which are closer than the ``capture`` distance
        are considered. Slave-points beyond the ends of a segment are not in contact
        with that segment (the contact is not artificially extended along the segment
        tangent) and slave-points which are too far away from the segment are ignored
        (this avoids spurious contact of points located on the inner side of the
        infinite plane of a distant segment).
        """

        best = None

        for facet, corners in enumerate(corners_all):
            corners_coords = self.points[corners] + displacement[corners]
            result = self._project(xs, corners_coords, normals_ref[facet])

            if result is None or not result["inside"]:
                continue

            distance = np.linalg.norm(result["gap_vector"])

            if distance > capture:
                continue

            result["cells"] = corners
            result["normal_ref"] = normals_ref[facet]

            # nearest facet by the true (Euclidean) distance to the projection point
            if best is None or distance < best["distance"]:
                result["distance"] = distance
                best = result

        return best

    @staticmethod
    def _bilinear_batch(xi):
        "Batched bilinear quad shape-functions and their derivatives."

        a = xi[..., 0]
        b = xi[..., 1]
        shape = 0.25 * np.stack(
            [(1 - a) * (1 - b), (1 + a) * (1 - b), (1 + a) * (1 + b), (1 - a) * (1 + b)],
            axis=-1,
        )
        dshape = 0.25 * np.stack(
            [
                np.stack([-(1 - b), (1 - b), (1 + b), -(1 + b)], axis=-1),
                np.stack([-(1 - a), -(1 + a), (1 + a), (1 - a)], axis=-1),
            ],
            axis=-2,
        )
        return shape, dshape

    def _closest_batch_2d(self, nodes, corners_all, normals_ref, displacement, capture):
        tol = 1e-4
        ns = len(nodes)
        nc = corners_all.shape[1]
        zeros_cells = np.zeros((ns, nc), dtype=int)
        zeros_shape = np.zeros((ns, nc), dtype=float)
        zeros_vec = np.zeros((ns, self.dim), dtype=float)

        if ns == 0 or len(corners_all) == 0:
            return {
                "valid": np.zeros(ns, dtype=bool),
                "cells": zeros_cells,
                "normal": zeros_vec,
                "shape": zeros_shape,
                "gap": np.zeros(ns, dtype=float),
                "gap_vector": zeros_vec.copy(),
                "normal_ref": zeros_vec.copy(),
                "xs": zeros_vec.copy(),
            }

        xs = self.points[nodes] + displacement[nodes]
        corners_coords = self.points[corners_all] + displacement[corners_all]
        a = corners_coords[:, 0]
        b = corners_coords[:, 1]
        tangent = b - a
        length_squared = np.einsum("fd,fd->f", tangent, tangent)
        valid_facet = length_squared > 0.0

        denom = np.where(valid_facet, length_squared, 1.0)
        diff = xs[:, None, :] - a[None, :, :]
        xi = np.einsum("sfd,fd->sf", diff, tangent) / denom[None, :]
        inside = (xi >= -tol) & (xi <= 1.0 + tol) & valid_facet[None, :]
        xi_clipped = np.clip(xi, 0.0, 1.0)

        projection = a[None, :, :] + xi_clipped[..., None] * tangent[None, :, :]
        normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        normals = normals / np.sqrt(denom)[:, None]
        normals = np.where(
            (np.einsum("fd,fd->f", normals, normals_ref) < 0)[:, None], -normals, normals
        )

        gap_vector = xs[:, None, :] - projection
        gap = np.einsum("sfd,fd->sf", gap_vector, normals)
        distance = np.linalg.norm(gap_vector, axis=2)
        valid = inside & (distance <= capture)

        best = np.argmin(np.where(valid, distance, np.inf), axis=1)
        has = np.any(valid, axis=1)
        idx = np.arange(ns)

        cells = zeros_cells.copy()
        shape = zeros_shape.copy()
        normal = zeros_vec.copy()
        gap_selected = np.zeros(ns, dtype=float)
        gap_vector_selected = zeros_vec.copy()
        normal_ref_selected = zeros_vec.copy()

        if np.any(has):
            sel = has
            cells[sel] = corners_all[best[sel]]
            xi_sel = xi_clipped[idx[sel], best[sel]]
            shape[sel] = np.column_stack([1.0 - xi_sel, xi_sel])
            normal[sel] = normals[best[sel]]
            gap_selected[sel] = gap[idx[sel], best[sel]]
            gap_vector_selected[sel] = gap_vector[idx[sel], best[sel]]
            normal_ref_selected[sel] = normals_ref[best[sel]]

        return {
            "valid": has,
            "cells": cells,
            "normal": normal,
            "shape": shape,
            "gap": gap_selected,
            "gap_vector": gap_vector_selected,
            "normal_ref": normal_ref_selected,
            "xs": xs,
        }

    def _closest_batch_3d(self, nodes, corners_all, normals_ref, displacement, capture):
        tol = 1e-4
        ns = len(nodes)
        nf = len(corners_all)
        nc = corners_all.shape[1]
        zeros_cells = np.zeros((ns, nc), dtype=int)
        zeros_shape = np.zeros((ns, nc), dtype=float)
        zeros_vec = np.zeros((ns, self.dim), dtype=float)

        if ns == 0 or nf == 0:
            return {
                "valid": np.zeros(ns, dtype=bool),
                "cells": zeros_cells,
                "normal": zeros_vec,
                "shape": zeros_shape,
                "gap": np.zeros(ns, dtype=float),
                "gap_vector": zeros_vec.copy(),
                "normal_ref": zeros_vec.copy(),
                "xs": zeros_vec.copy(),
            }

        xs = self.points[nodes] + displacement[nodes]
        corners_coords = self.points[corners_all] + displacement[corners_all]

        xi = np.zeros((ns, nf, 2), dtype=float)
        pair_valid = np.ones((ns, nf), dtype=bool)

        for _ in range(8):
            shape, dshape = self._bilinear_batch(xi)
            projection = np.einsum("sfk,fkd->sfd", shape, corners_coords)
            t1 = np.einsum("sfk,fkd->sfd", dshape[..., 0, :], corners_coords)
            t2 = np.einsum("sfk,fkd->sfd", dshape[..., 1, :], corners_coords)
            residual = np.stack(
                [
                    np.einsum("sfd,sfd->sf", xs[:, None, :] - projection, t1),
                    np.einsum("sfd,sfd->sf", xs[:, None, :] - projection, t2),
                ],
                axis=-1,
            )

            t1t1 = np.einsum("sfd,sfd->sf", t1, t1)
            t1t2 = np.einsum("sfd,sfd->sf", t1, t2)
            t2t2 = np.einsum("sfd,sfd->sf", t2, t2)

            a11 = -t1t1
            a12 = -t1t2
            a21 = -t1t2
            a22 = -t2t2

            det = a11 * a22 - a12 * a21
            valid_det = np.abs(det) > 0.0
            pair_valid &= valid_det

            inv_det = np.where(valid_det, 1.0 / det, 0.0)
            dxi = np.zeros_like(residual)
            dxi[..., 0] = (residual[..., 0] * a22 - residual[..., 1] * a12) * inv_det
            dxi[..., 1] = (a11 * residual[..., 1] - a21 * residual[..., 0]) * inv_det
            xi -= dxi

        inside = np.all(np.abs(xi) <= 1.0 + tol, axis=-1) & pair_valid
        xi_clipped = np.clip(xi, -1.0, 1.0)
        shape, dshape = self._bilinear_batch(xi_clipped)
        projection = np.einsum("sfk,fkd->sfd", shape, corners_coords)
        t1 = np.einsum("sfk,fkd->sfd", dshape[..., 0, :], corners_coords)
        t2 = np.einsum("sfk,fkd->sfd", dshape[..., 1, :], corners_coords)

        normal = np.cross(t1, t2)
        norm = np.linalg.norm(normal, axis=2)
        valid_norm = norm > 0.0
        normal = np.divide(normal, norm[..., None], out=np.zeros_like(normal), where=valid_norm[..., None])

        orientation = np.einsum("sfd,fd->sf", normal, normals_ref)
        normal = np.where(orientation[..., None] < 0.0, -normal, normal)

        gap_vector = xs[:, None, :] - projection
        gap = np.einsum("sfd,sfd->sf", gap_vector, normal)
        distance = np.linalg.norm(gap_vector, axis=2)
        valid = inside & valid_norm & (distance <= capture)

        best = np.argmin(np.where(valid, distance, np.inf), axis=1)
        has = np.any(valid, axis=1)
        idx = np.arange(ns)

        cells = zeros_cells.copy()
        shape_selected = zeros_shape.copy()
        normal_selected = zeros_vec.copy()
        gap_selected = np.zeros(ns, dtype=float)
        gap_vector_selected = zeros_vec.copy()
        normal_ref_selected = zeros_vec.copy()

        if np.any(has):
            sel = has
            cells[sel] = corners_all[best[sel]]
            shape_selected[sel] = shape[idx[sel], best[sel]]
            normal_selected[sel] = normal[idx[sel], best[sel]]
            gap_selected[sel] = gap[idx[sel], best[sel]]
            gap_vector_selected[sel] = gap_vector[idx[sel], best[sel]]
            normal_ref_selected[sel] = normals_ref[best[sel]]

        return {
            "valid": has,
            "cells": cells,
            "normal": normal_selected,
            "shape": shape_selected,
            "gap": gap_selected,
            "gap_vector": gap_vector_selected,
            "normal_ref": normal_ref_selected,
            "xs": xs,
        }

    def _closest_batch(self, nodes, corners_all, normals_ref, displacement, capture):
        if self.dim == 2:
            return self._closest_batch_2d(nodes, corners_all, normals_ref, displacement, capture)
        return self._closest_batch_3d(nodes, corners_all, normals_ref, displacement, capture)

    def _friction(self, contact, dx_ref, eps_n, eps_t):
        """Evaluate the Coulomb friction state (stick/slip) and update the reference
        gap-vector by a return-mapping. Returns a tuple ``(slip, dx_ref)``.
        """

        normal = contact["normal"]
        projection = np.eye(self.dim) - np.outer(normal, normal)
        slip_vector = projection @ (contact["gap_vector"] - dx_ref)
        force_trial = eps_t * slip_vector

        gap = contact["gap"]
        force_normal_abs = eps_n * abs(gap)
        force_limit = self.friction * force_normal_abs
        force_norm = np.linalg.norm(force_trial)

        stick = True
        if np.isfinite(self.friction):
            tol = np.sqrt(np.finfo(float).eps) * max(force_limit, 1.0)
            stick = force_norm <= force_limit + tol

        dx_ref_new = np.array(dx_ref, dtype=float)
        eps = np.sqrt(np.finfo(float).eps) * max(force_normal_abs, 1.0)
        if (not stick) and force_norm > eps:
            force_t = (force_limit / force_norm) * force_trial
            dx_ref_new = contact["gap_vector"] - force_t / eps_t

        return (not stick), dx_ref_new

    def _local_force(self, coords, normal_ref, dx_ref, slip, eps_n, eps_t):
        """Evaluate the local contact force-vector for one slave-point and its master
        segment as a pure function of the (deformed) local coordinates. The first row of
        ``coords`` is the slave-point, the remaining rows are the master-segment nodes.

        The friction reference gap-vector ``dx_ref`` and the ``slip``-state are held
        fixed - this makes the finite-difference tangent consistent with the residual.
        """

        xs = coords[0]
        corners_coords = coords[1:]
        local = np.zeros_like(coords)

        result = self._project(xs, corners_coords, normal_ref)

        if result is None or result["gap"] >= 0.0:
            return local.ravel()

        normal = result["normal"]
        shape = result["shape"]
        gap = result["gap"]

        # normal penalty force on the slave-point and (distributed) on the master-nodes
        force = eps_n * gap * normal
        local[0] += force
        local[1:] -= shape[:, None] * force

        # tangential Coulomb friction force
        if self.friction > 0.0:
            projection = np.eye(self.dim) - np.outer(normal, normal)
            slip_vector = projection @ (result["gap_vector"] - dx_ref)
            force_trial = eps_t * slip_vector

            force_t = force_trial
            if slip:  # scale the trial force to the friction limit (return-mapping)
                force_normal_abs = eps_n * abs(gap)
                force_limit = self.friction * force_normal_abs
                force_norm = np.linalg.norm(force_trial)
                eps = np.sqrt(np.finfo(float).eps) * max(force_normal_abs, 1.0)
                if force_norm > eps:
                    force_t = (force_limit / force_norm) * force_trial

            local[0] += force_t
            local[1:] -= shape[:, None] * force_t

        return local.ravel()

    def _vector(self, field=None, parallel=False):
        "Assemble the sparse residual force-vector of the contact contributions."

        if field is not None:
            self.field = field

        displacement = self.field.fields[0].values
        force = np.zeros_like(displacement, dtype=float)
        eps_n, eps_t = self._penalties()

        for contact_pass, state in zip(self._pass_cache, self._states):
            nodes = contact_pass["nodes"]
            contact = self._closest_batch(
                nodes,
                contact_pass["corners"],
                contact_pass["normals_ref"],
                displacement,
                contact_pass["capture"],
            )
            contact_mask = contact["valid"] & (contact["gap"] < 0.0)

            new = contact_mask & (~state["active"])
            state["dx_ref"][new] = contact["gap_vector"][new]

            state["active"][:] = contact_mask
            state["slip"][~contact_mask] = False

            if not np.any(contact_mask):
                continue

            active = np.where(contact_mask)[0]
            active_nodes = nodes[active]
            gap_active = contact["gap"][active]
            normal_active = contact["normal"][active]
            gap_vector_active = contact["gap_vector"][active]

            force_contact = eps_n * gap_active[:, None] * normal_active

            if self.friction > 0.0:
                projection = np.eye(self.dim)[None, :, :] - np.einsum(
                    "ai,aj->aij", normal_active, normal_active
                )
                slip_vector = np.einsum(
                    "aij,aj->ai",
                    projection,
                    gap_vector_active - state["dx_ref"][active],
                )
                force_trial = eps_t * slip_vector
                force_normal_abs = eps_n * np.abs(gap_active)
                force_limit = self.friction * force_normal_abs
                force_norm = np.linalg.norm(force_trial, axis=1)

                stick = np.ones(len(active), dtype=bool)
                if np.isfinite(self.friction):
                    tol = np.sqrt(np.finfo(float).eps) * np.maximum(force_limit, 1.0)
                    stick = force_norm <= (force_limit + tol)

                force_t = force_trial.copy()
                eps = np.sqrt(np.finfo(float).eps) * np.maximum(force_normal_abs, 1.0)
                slide = (~stick) & (force_norm > eps)
                if np.any(slide):
                    force_t[slide] = (
                        force_limit[slide] / force_norm[slide]
                    )[:, None] * force_trial[slide]
                    active_slide = active[slide]
                    state["dx_ref"][active_slide] = (
                        gap_vector_active[slide] - force_t[slide] / eps_t
                    )

                state["slip"][active] = ~stick
                force_contact += force_t
            else:
                state["slip"][active] = False

            np.add.at(force, active_nodes, force_contact)
            master_force = -contact["shape"][active, :, None] * force_contact[:, None, :]
            np.add.at(force, contact["cells"][active].ravel(), master_force.reshape(-1, self.dim))

        self.results.force = csr_matrix(force.reshape(-1, 1))

        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Assemble the sparse tangent stiffness-matrix of the contact contributions."

        if field is not None:
            self.field = field

        displacement = self.field.fields[0].values
        indices = self.field[0].indices.dof
        eps_n, eps_t = self._penalties()
        rows = []
        cols = []
        data = []

        for contact_pass, state in zip(self._pass_cache, self._states):
            nodes = contact_pass["nodes"]
            contact = self._closest_batch(
                nodes,
                contact_pass["corners"],
                contact_pass["normals_ref"],
                displacement,
                contact_pass["capture"],
            )
            contact_mask = contact["valid"] & (contact["gap"] < 0.0)

            if not np.any(contact_mask):
                continue

            for i in np.where(contact_mask)[0]:
                node = nodes[i]
                corners = contact["cells"][i]
                coords = np.vstack(
                    [contact["xs"][i], self.points[corners] + displacement[corners]]
                )

                # consistent local tangent by finite-differences of the local force,
                # with the friction reference state held fixed
                flat = coords.ravel().astype(float)
                force0 = self._local_force(
                    coords,
                    contact["normal_ref"][i],
                    state["dx_ref"][i],
                    state["slip"][i],
                    eps_n,
                    eps_t,
                )
                nloc = flat.size
                Kloc = np.zeros((nloc, nloc))
                h = 1e-7 * max(1.0, np.abs(flat).max())
                for k in range(nloc):
                    perturbed = flat.copy()
                    perturbed[k] += h
                    force1 = self._local_force(
                        perturbed.reshape(-1, self.dim),
                        contact["normal_ref"][i],
                        state["dx_ref"][i],
                        state["slip"][i],
                        eps_n,
                        eps_t,
                    )
                    Kloc[:, k] = (force1 - force0) / h

                participants = np.concatenate([[node], corners])
                dof = indices[participants].ravel()
                rows.append(np.repeat(dof, nloc))
                cols.append(np.tile(dof, nloc))
                data.append(Kloc.ravel())

        if rows:
            self.results.stiffness = coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.mesh.ndof, self.mesh.ndof),
            ).tocsr()
        else:
            self.results.stiffness = csr_matrix((self.mesh.ndof, self.mesh.ndof))

        return self.results.stiffness

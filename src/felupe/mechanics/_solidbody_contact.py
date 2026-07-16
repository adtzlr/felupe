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
from scipy.sparse import lil_matrix

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
        r = lil_matrix(displacement.shape)
        eps_n, eps_t = self._penalties()

        for (nodes, corners_all, normals_ref, capture), state in zip(
            self._passes, self._states
        ):
            for i, node in enumerate(nodes):
                xs = self.points[node] + displacement[node]
                contact = self._closest(
                    xs, corners_all, normals_ref, displacement, capture
                )

                if contact is None or contact["gap"] >= 0.0:
                    state["active"][i] = False
                    continue

                if not state["active"][i]:  # newly closed contact
                    state["dx_ref"][i] = contact["gap_vector"]

                state["active"][i] = True

                # update the friction state (stick/slip) and reference gap-vector
                if self.friction > 0.0:
                    state["slip"][i], state["dx_ref"][i] = self._friction(
                        contact, state["dx_ref"][i], eps_n, eps_t
                    )

                corners = contact["cells"]
                coords = np.vstack([xs, self.points[corners] + displacement[corners]])

                # evaluate the local force-vector (shared with the tangent assembly)
                local = self._local_force(
                    coords,
                    contact["normal_ref"],
                    state["dx_ref"][i],
                    state["slip"][i],
                    eps_n,
                    eps_t,
                ).reshape(-1, self.dim)

                r[node] += local[0]
                for a, corner in enumerate(corners):
                    r[corner] += local[1 + a]

        self.results.force = r.reshape(-1, 1).tocsr()

        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Assemble the sparse tangent stiffness-matrix of the contact contributions."

        if field is not None:
            self.field = field

        displacement = self.field.fields[0].values
        indices = self.field[0].indices.dof
        K = lil_matrix((self.mesh.ndof, self.mesh.ndof))
        eps_n, eps_t = self._penalties()

        for (nodes, corners_all, normals_ref, capture), state in zip(
            self._passes, self._states
        ):
            for i, node in enumerate(nodes):
                xs = self.points[node] + displacement[node]
                contact = self._closest(
                    xs, corners_all, normals_ref, displacement, capture
                )

                if contact is None or contact["gap"] >= 0.0:
                    continue

                corners = contact["cells"]
                coords = np.vstack([xs, self.points[corners] + displacement[corners]])

                # consistent local tangent by finite-differences of the local force,
                # with the friction reference state held fixed
                flat = coords.ravel().astype(float)
                force0 = self._local_force(
                    coords,
                    contact["normal_ref"],
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
                        contact["normal_ref"],
                        state["dx_ref"][i],
                        state["slip"][i],
                        eps_n,
                        eps_t,
                    )
                    Kloc[:, k] = (force1 - force0) / h

                # scatter the local tangent into the global sparse matrix
                participants = np.concatenate([[node], corners])
                dof = indices[participants].ravel()
                K[dof.reshape(-1, 1), dof] += Kloc

        self.results.stiffness = K.tocsr()

        return self.results.stiffness

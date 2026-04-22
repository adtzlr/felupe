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


class ContactPlane:

    def plot(
        self,
        plotter=None,
        offset=0.0,
        show_edges=True,
        color="black",
        opacity=0.5,
        deformed=True,
        size=None,
        line_width=None,
        show_point=True,
        show_line=True,
        sym=(False, False, False),
        **kwargs,
    ):
        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()

        dim = self.mesh.dim
        x = self.mesh.points

        if deformed:
            x = self.mesh.points + self.field[0].values

        center = np.pad(x[self.centerpoint] - offset * self.normal, (0, 3 - dim))
        normal = np.pad(self.normal, (0, 3 - dim))

        dx = (x.max(axis=0) - x.min(axis=0)).max()

        if size is None:
            size = 1.05 * dx

        plane = pv.Plane(
            center=center,
            direction=normal,
            i_resolution=1,
            j_resolution=1,
            i_size=size,
            j_size=size,
        )

        axes = ["x", "y", "z"]
        for ax in np.where(np.array(sym, dtype=bool))[0]:

            invert = False
            if np.any(self.mesh.points[:, ax] < 0):
                invert = True

            plane = plane.clip(axes[ax], invert=invert, origin=(0.0, 0.0, 0.0))

        if dim == 2:
            z = plane.points[:, 2]
            points = plane.points[z == z.min()]
            points[:, 2] = 0
            plane = pv.Line(*points)

        plotter.add_mesh(
            plane,
            show_edges=show_edges,
            opacity=opacity,
            color=color,
            line_width=line_width,
            **kwargs,
        )

        if show_point:
            plotter.add_points(
                center.reshape(1, -1),
                opacity=opacity,
                color=color,
                point_size=12,
                **kwargs,
            )

        if show_line:
            plotter.add_mesh(
                pv.Line(center, center - 0.1 * dx * normal),
                opacity=opacity,
                color=color,
                line_width=line_width,
                **kwargs,
            )

        return plotter


class ContactRigidPlane(ContactPlane):
    r"""A node-to-surface contact, where the surface is given by a rigid plane.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray
        An array with indices of points to be connected to the center-point.
    centerpoint : int
        The index of the centerpoint.
    normal : ndarray
        The outward plane normal vector.
    friction : float, optional
        Coulomb friction coefficient :math:`\mu`. Default is 0.0.
    multiplier : float, optional
        A multiplier to penalize the relative displacements between the center-point and
        the points in contact. Default is 1e6.

    Notes
    -----
    A :class:`~felupe.ContactRigidPlane` is supported as an item in a
    :class:`~felupe.Step`.

    ..  note::

        The contact formulation is based on a penalty method. The multiplier should be
        chosen sufficiently large to enforce the contact constraints, but not too large
        to cause numerical issues. Furthermore, no regularization is applied, which may
        lead to convergence issues when the contact status changes. Frictionless and
        nearly sticking contact conditions are supported.

    The contact activation is based on the gap between the center-point and the points
    in potential contact. The gap is evaluated in the deformed configuration in the
    direction of the plane normal. If the gap is negative, the contact is active, see
    Eq. :eq:`gap-vector`.

    ..  math::
        :label: gap-vector

        \Delta \boldsymbol{x} &= \boldsymbol{x} - \boldsymbol{x}_c

        g &= \Delta \boldsymbol{x} \cdot \boldsymbol{n}

        g &\lt 0 \quad \text{(contact active)}
    
    The contact normal force and tangent stiffness matrix are evaluated as a penalty
    contribution proportional to the gap, see Eq. :eq:`contact-force`.

    ..  math::
        :label: contact-force    
    
        \boldsymbol{f} &= \lambda\ g\ \boldsymbol{n}

        \boldsymbol{P}_n &= \boldsymbol{n} \otimes \boldsymbol{n}

        \boldsymbol{K}_n &= \lambda\ \boldsymbol{P}_n

    The tangential contact friction forces are evaluated according to a Coulomb friction
    law, see Eq. :eq:`contact-friction` and Eq.
    :eq:`contact-friction-state`.

    ..  math::
        :label: contact-friction

        \Delta \boldsymbol{x}_t &= \Delta \boldsymbol{x} - \Delta \boldsymbol{x}^{ref}

        \boldsymbol{P}_t &= \boldsymbol{1} - \boldsymbol{P}_n

        \boldsymbol{f}_t^{trial} &= \lambda \boldsymbol{P}_t\ \Delta \boldsymbol{x}_t

        f_t^{limit} &= \mu |\boldsymbol{f}_n|

    ..  math::
        :label: contact-friction-state

        \text{state} = \begin{cases}
            |\boldsymbol{f}_t^{trial}| \leq f_t^{limit} & \text{stick} \\
            \text{else} & \text{slip}
        \end{cases}

    In case of sticking contact, the tangential forces are equal to the trial forces,
    see Eq. :eq:`contact-friction-stick`.

    ..  math::
        :label: contact-friction-stick

        \boldsymbol{f}_t &= \boldsymbol{f}_t^{trial}

        \boldsymbol{K}_t &= \lambda\ \boldsymbol{P}_t

    In case of sliding contact, a scale factor is applied to the tangential forces to
    enforce the friction limit, see Eq. :eq:`contact-friction-slide`.

    ..  math::
        :label: contact-friction-slide

        s_t &= \frac{f_t^{limit}}{|\boldsymbol{f}_t^{trial}|}

        \boldsymbol{f}_t &= s_t\ \boldsymbol{f}_t^{trial}

        \Delta \boldsymbol{x}^{ref} &=
            \Delta \boldsymbol{x} - \frac{\boldsymbol{f}_t}{\lambda}
    
    The tangential stiffness matrix contribution for sliding contact is given in Eq.
    :eq:`contact-friction-slide-stiffness`.

    ..  math::
        :label: contact-friction-slide-stiffness

        \hat{\boldsymbol{f}_t} &= \frac{
            \boldsymbol{f}_t^{trial}
        }{|\boldsymbol{f}_t^{trial}|}

        \hat{\boldsymbol{P}}_t &= \frac{1}{|\boldsymbol{f}_t^{trial}|} \left(
            \boldsymbol{1} - \hat{\boldsymbol{f}}_t \otimes \hat{\boldsymbol{f}}_t
        \right)
        
        \boldsymbol{K}_t &=
            \mu\ \lambda\ |\boldsymbol{f}_n|\ \hat{\boldsymbol{P}}_t\ \boldsymbol{P}_t

    Examples
    --------
    This example shows how to use a :class:`~felupe.ContactRigidPlane`.

    An additional center-point is added to a mesh. By default, all *hanging* points are
    collected in the mesh-attribute
    :attr:`Mesh.points_without_cells <felupe.Mesh.points_without_cells>`. The degrees of
    freedom of these points are considered as fixed, i.e. they are ignored. The center-
    point is not connected to any cell and is added to the points-without-cells array
    on :meth:`Mesh.update <felupe.Mesh.update>`. Hence, the center-point has to be
    removed manually.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> mesh.update(points=np.vstack([mesh.points, [2.0, 0.5, 0.5]]))
        >>>
        >>> # prevent the field-values at the center-point to be treated as dof0
        >>> mesh.points_without_cells = mesh.points_without_cells[:-1]
        >>>
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>>
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> solid = fem.SolidBody(umat=umat, field=field)

    A :class:`~felupe.ContactRigidPlane` defines the contact, which connects the
    displacement degrees of freedom of the center-point with the dofs of points located
    at :math:`x=1` if they are in contact. Only the :math:`x`-component is considered in
    this example.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> contact = fem.ContactRigidPlane(
        ...     field=field,
        ...     points=np.arange(mesh.npoints)[mesh.x == 1],
        ...     centerpoint=-1,
        ...     normal=[-1.0, 0, 0],
        ...     friction=0.5,
        ...     multiplier=1e2,
        ... )
        >>>
        >>> plotter = pv.Plotter()
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[contact.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[contact.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> mesh.plot(plotter=contact.plot(size=1.2, plotter=plotter)).show()

    The mesh is fixed on the left end face and a ramped :class:`~felupe.Boundary` is
    applied on the center-point of the :class:`~felupe.ContactRigidPlane`. All items
    are added to a :class:`~felupe.Step` and a :class:`~felupe.Job` is evaluated.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {
        ...     "fixed": fem.Boundary(displacement, fx=0),
        ...     "control": fem.Boundary(displacement, fx=2, skip=(1, 0, 0)),
        ...     "move": fem.Boundary(displacement, fx=2, skip=(0, 1, 1)),
        ... }
        >>> table = fem.math.linsteps([0, -1, -1.5], num=5)
        >>> step = fem.Step(
        ...     [solid, contact],
        ...     boundaries=boundaries,
        ...     ramp={boundaries["move"]: table},
        ... )
        >>> job = fem.Job([step]).evaluate()

    A view on the deformed mesh including the :class:`~felupe.ContactRigidPlane` is
    plotted.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> plotter = pv.Plotter()
        >>>
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[contact.points] + displacement.values[contact.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[contact.centerpoint]] + displacement.values[[contact.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> field.plot(
        ...     "Displacement", 
        ...     component=None, 
        ...     show_undeformed=False, 
        ...     plotter=contact.plot(size=1.2, plotter=plotter),
        ...     clim=[0, 0.5],
        ... ).show()

    """

    def __init__(
        self,
        field,
        points,
        centerpoint,
        normal,
        friction=0.0,
        multiplier=1e6,
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.array(points, dtype=int)
        self.centerpoint = centerpoint

        self.normal = np.array(normal, dtype=float)[: self.mesh.dim]
        self.normal /= np.linalg.norm(self.normal)

        # normal and tangential projections
        self.projection_normal = np.outer(self.normal, self.normal)
        self.projection_tangential = np.eye(self.mesh.dim) - self.projection_normal

        self.friction = friction
        self.multiplier = multiplier
        self.multiplier_tangential = multiplier * 0.1

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.results = Results(stress=False, elasticity=False)

        self.results.dx_ref = np.zeros_like(self.field.fields[0].values[self.points])
        self.results.active = np.zeros(len(self.points), dtype=bool)
        self.results.slip = np.zeros(len(self.points), dtype=bool)

        # differences of undeformed coordinates between points in potential contact
        # and center-point (initial gap vectors)
        self.dX = self.mesh.points[self.points] - self.mesh.points[self.centerpoint]

    def _vector(self, field=None, parallel=False):

        if field is not None:
            self.field = field

        # displacement field values at mesh points
        u = self.field.fields[0].values

        # gap vectors
        dx = self.dX + (u[self.points] - u[self.centerpoint])
        gap = dx @ self.normal
        contact_mask = gap < 0.0

        # new points in contact
        new = contact_mask & (~self.results.active)
        self.results.dx_ref[new] = dx[new]

        # update active contact mask
        self.results.active = contact_mask.copy()
        contact = np.where(contact_mask)[0]

        r = lil_matrix((self.mesh.ndof, self.mesh.dim))

        if len(contact) > 0:

            # normal contribution
            force_n = self.multiplier * np.outer(gap[contact], self.normal)

            r[self.points[contact]] += force_n
            r[self.centerpoint] -= force_n.sum(axis=0)

            if self.friction > 0.0:  # tangential Coulomb friction contribution
                dx_delta = dx[contact] - self.results.dx_ref[contact]
                slip_t = dx_delta @ self.projection_tangential
                force_t_trial = self.multiplier_tangential * slip_t

                force_n_abs = self.multiplier * np.abs(gap[contact])
                force_t_limit = self.friction * force_n_abs
                force_t_norm = np.linalg.norm(force_t_trial, axis=1)

                # points with friction: sticking or sliding
                stick = np.ones(len(contact), dtype=bool)
                if np.isfinite(self.friction):
                    tol = np.sqrt(np.finfo(float).eps) * np.maximum(force_t_limit, 1.0)
                    stick = force_t_norm <= (force_t_limit + tol)

                force_t = force_t_trial.copy()
                eps = np.sqrt(np.finfo(float).eps) * np.maximum(force_n_abs, 1.0)
                slide = (~stick) & (force_t_norm > eps)
                if np.any(slide):
                    scale = force_t_limit[slide] / force_t_norm[slide]
                    force_t[slide] = scale.reshape(-1, 1) * force_t_trial[slide]

                    # return mapping for sliding points
                    contact_slide = contact[slide]
                    self.results.dx_ref[contact_slide] = (
                        dx[contact_slide] - force_t[slide] / self.multiplier_tangential
                    )

                self.results.slip[contact] = ~stick

                r[self.points[contact]] += force_t
                r[self.centerpoint] -= force_t.sum(axis=0)

        self.results.force = r.reshape(-1, 1).tocsr()

        return self.results.force

    def _matrix(self, field=None, parallel=False):

        if field is not None:
            self.field = field

        # displacement field values at mesh points
        u = self.field.fields[0].values

        # gap vectors
        dx = self.dX + (u[self.points] - u[self.centerpoint])
        gap = dx @ self.normal
        contact_mask = gap < 0.0
        contact = np.where(contact_mask)[0]

        indices = self.field[0].indices.dof
        K = lil_matrix((self.mesh.ndof, self.mesh.ndof))

        # normal stiffness contribution
        if len(contact) > 0:

            idx = indices[self.points[contact]]
            ctr = indices[self.centerpoint]

            dim = self.mesh.dim
            npoints = len(contact) * dim

            # evaluate normal stiffness: λ n ⊗ n
            K_n = self.multiplier * self.projection_normal

            K_n_pp = np.einsum("ab,ij->aibj", np.eye(len(contact)), K_n).reshape(
                npoints, npoints
            )

            K_n_pc = np.broadcast_to(
                K_n, (len(contact), self.mesh.dim, self.mesh.dim)
            ).reshape(npoints, dim)

            K_n_cc = K_n_pc.sum(axis=0)

            K[idx.reshape(-1, 1), idx.ravel()] += K_n_pp
            K[idx.reshape(-1, 1), ctr.ravel()] -= K_n_pc
            K[ctr.reshape(-1, 1), idx.ravel()] -= K_n_pc.T
            K[ctr.reshape(-1, 1), ctr.ravel()] += K_n_cc

            # tangential stiffness contribution (stick)
            if self.friction > 0.0 and np.any(~self.results.slip[contact]):
                contact_stick = contact[~self.results.slip[contact]]

                idx = indices[self.points[contact_stick]]
                ctr = indices[self.centerpoint]

                dim = self.mesh.dim
                npoints = len(contact_stick) * dim

                # evaluate tangential stiffness: λ (1 - n ⊗ n)
                K_t = self.multiplier_tangential * self.projection_tangential

                K_t_pp = np.einsum(
                    "ab,ij->aibj", np.eye(len(contact_stick)), K_t
                ).reshape(npoints, npoints)

                K_t_pc = np.broadcast_to(
                    K_t,
                    (len(contact_stick), self.mesh.dim, self.mesh.dim),
                ).reshape(npoints, dim)
                K_t_cc = K_t_pc.sum(axis=0)

                K[idx.reshape(-1, 1), idx.ravel()] += K_t_pp
                K[idx.reshape(-1, 1), ctr.ravel()] -= K_t_pc
                K[ctr.reshape(-1, 1), idx.ravel()] -= K_t_pc.T
                K[ctr.reshape(-1, 1), ctr.ravel()] += K_t_cc

            # tangential stiffness contribution (slip)
            if self.friction > 0.0 and np.any(self.results.slip[contact]):
                contact_slip = contact[self.results.slip[contact]]

                idx = indices[self.points[contact_slip]]
                ctr = indices[self.centerpoint]

                dim = self.mesh.dim
                npoints = len(contact_slip) * dim

                dx_delta = dx[contact_slip] - self.results.dx_ref[contact_slip]
                slip_t = dx_delta @ self.projection_tangential
                force_t_trial = self.multiplier_tangential * slip_t

                force_n_abs = self.multiplier * np.abs(gap[contact_slip])
                force_t_norm = np.linalg.norm(force_t_trial, axis=1)

                t = force_t_trial
                norm_t = force_t_norm

                eps = np.sqrt(np.finfo(float).eps) * np.maximum(force_n_abs, 1.0)
                mask = norm_t > eps
                t_hat = np.zeros_like(t)
                t_hat[mask] = t[mask] / norm_t[mask, None]

                projection_slip = np.zeros((len(contact_slip), dim, dim))
                projection_slip[mask] = (
                    np.eye(dim)[None, :, :]
                    - np.einsum("ai,aj->aij", t_hat[mask], t_hat[mask])
                ) / norm_t[mask, None, None]

                # evaluate tangential stiffness (slip)
                K_t = (
                    self.friction
                    * force_n_abs[:, None, None]
                    * projection_slip
                    @ (self.multiplier_tangential * self.projection_tangential)
                ) - (
                    self.friction
                    * self.multiplier
                    * np.einsum("ai,j->aij", t_hat, self.normal)
                )

                K_t_pp = np.einsum(
                    "ab,aij->aibj", np.eye(len(contact_slip)), K_t
                ).reshape(npoints, npoints)

                K_t_pc = np.broadcast_to(
                    K_t,
                    (len(contact_slip), self.mesh.dim, self.mesh.dim),
                ).reshape(npoints, dim)

                K_t_cc = K_t_pc.sum(axis=0)

                K[idx.reshape(-1, 1), idx.ravel()] += K_t_pp
                K[idx.reshape(-1, 1), ctr.ravel()] -= K_t_pc
                K[ctr.reshape(-1, 1), idx.ravel()] -= K_t_pc.T
                K[ctr.reshape(-1, 1), ctr.ravel()] += K_t_cc

        self.results.stiffness = K.tocsr()

        return self.results.stiffness

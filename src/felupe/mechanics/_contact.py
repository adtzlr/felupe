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

from collections import namedtuple

import numpy as np
from scipy.sparse import coo_matrix

from ._helpers import Assemble, Results

ContactMultipliers = namedtuple("ContactMultipliers", ["normal", "tangential"])


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
        line_width_normal=None,
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
                line_width=line_width_normal,
                **kwargs,
            )

        return plotter


ContactTrial = namedtuple(
    "ContactTrial", ["contact", "force", "stiffness", "dx", "dx_ref", "active", "slip"]
)


def _assemble_vector(dof, dof_center, forces, size, dtype):
    "Assemble the forces (m, dim) of the points and the reaction of the center-point."

    rows = np.concatenate([dof.ravel(), dof_center.ravel()])
    data = np.concatenate([forces.ravel(), -forces.sum(axis=0)])
    cols = np.zeros_like(rows)

    return coo_matrix((data, (rows, cols)), shape=(size, 1), dtype=dtype).tocsr()


def _assemble_matrix(dof, dof_center, blocks, size, dtype):
    """Assemble the stiffness blocks (m, dim, dim) of the points, which are connected
    to the center-point."""

    m, dim, _ = blocks.shape
    center = np.broadcast_to(dof_center.ravel(), (m, dim))

    rows, cols, data = [], [], []
    for row, col, sign in [
        (dof, dof, 1),
        (dof, center, -1),
        (center, dof, -1),
        (center, center, 1),
    ]:
        rows.append(np.broadcast_to(row[:, :, None], blocks.shape).ravel())
        cols.append(np.broadcast_to(col[:, None, :], blocks.shape).ravel())
        data.append(sign * blocks.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    return coo_matrix((data, (rows, cols)), shape=(size, size), dtype=dtype).tocsr()


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
        A scale factor for the multiplier to penalize the relative displacements between
        the center-point and the points in contact. Default is 1.0. The final multiplier
        is scaled times a base multiplier, based on the mean normal stiffness, if items
        are given.
    multiplier_tangential : float or None, optional
        A scale factor for the multiplier to penalize the relative tangential
        displacements between the center-point and the points in contact. Default is
        0.1. The final multiplier is scaled times a base multiplier, based on the mean
        tangential stiffness, if items are given.
    items : list or None, optional
        A list of items to be considered for the evaluation of the mean stiffness-based
        multipliers. If None, the multipliers are not based on the mean stiffness.
        Default is None.
    multiplier_update : str, optional
        The update of the mean stiffness-based multipliers, either ``"increment"``
        (once per increment, based on the stiffness of the last converged increment),
        ``"iteration"`` (in each evaluation, i.e. in each Newton-iteration, without
        linearization) or ``"once"`` (only on the first evaluation). Default is
        ``"increment"``. Without ``items``, the multipliers are constant.
    symmetric : bool, optional
        A flag to return the symmetric part of the stiffness matrix. The consistent
        stiffness matrix of frictional contact is not symmetric. Default is False.
    elastic_slip : float or None, optional
        The allowable elastic slip :math:`\gamma` (a length) of sticking points. If
        given, the tangential multiplier is proportional to the normal force,
        :math:`\lambda_t = \mu\ f_n / \gamma`, i.e. a sticking point starts to slide
        at an elastic slip of :math:`\gamma`, independent of the contact pressure.
        The scale factor ``multiplier_tangential`` is not used. Typical values are 0.5 %
        to 1 % of a characteristic element length of the contact surface. For an
        infinite friction coefficient, :math:`\mu=1` is used for the tangential
        multiplier. If None, the tangential multiplier is constant. Default is None.

    Notes
    -----
    A :class:`~felupe.ContactRigidPlane` is supported as an item in a
    :class:`~felupe.Step`.

    ..  note::

        The contact formulation is based on a penalty method. The multiplier should be
        chosen sufficiently large to enforce the contact constraints, but not too large
        to cause numerical issues. Best practice is to provide ``items``, which are
        connected to the contact points. This will enable mean stiffness-based
        multipliers.

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

        \boldsymbol{f}_n &= \lambda_n\ g\ \boldsymbol{n}, \quad
            f_n = |\boldsymbol{f}_n| = -\lambda_n\ g

        \boldsymbol{P}_n &= \boldsymbol{n} \otimes \boldsymbol{n}

        \boldsymbol{K}_n &= \lambda_n\ \boldsymbol{P}_n

    The tangential contact friction forces are evaluated according to a Coulomb friction
    law with an elastic (penalty) stick and a return mapping, similar to
    elasto-plasticity. The state variables are the gap vectors
    :math:`\Delta \boldsymbol{x}_n`, the reference (stick) gap vectors
    :math:`\Delta \boldsymbol{x}^{ref}_n` and the contact flags of the last converged
    increment. These are only updated after a converged increment, i.e. the forces and
    the stiffness matrix are functions of the displacements within an increment. For
    points which are already in contact at the beginning of the increment, the
    (trial) elastic slip is given in Eq. :eq:`contact-friction`.

    ..  math::
        :label: contact-friction

        \boldsymbol{P}_t &= \boldsymbol{1} - \boldsymbol{P}_n

        \boldsymbol{e} &= \boldsymbol{P}_t
            \left( \Delta \boldsymbol{x} - \Delta \boldsymbol{x}^{ref}_n \right),
            \quad \boldsymbol{E} = \frac{\partial \boldsymbol{e}}{
            \partial \Delta \boldsymbol{x}} = \boldsymbol{P}_t

    For points which come into contact within the increment, the reference gap vector
    is located at the (linearly interpolated) point of first contact, where the gap is
    zero, see Eq. :eq:`contact-friction-new`. This avoids friction forces from the
    tangential movement before the contact is established, also for large increments.

    ..  math::
        :label: contact-friction-new

        s &= \frac{-g}{g_n - g}, \quad \boldsymbol{e} = s\ \boldsymbol{P}_t
            \left( \Delta \boldsymbol{x} - \Delta \boldsymbol{x}_n \right)

        \boldsymbol{E} &= s\ \boldsymbol{P}_t - \frac{g_n}{(g_n - g)^2}\
            \boldsymbol{P}_t \left( \Delta \boldsymbol{x} - \Delta \boldsymbol{x}_n
            \right) \otimes \boldsymbol{n}

    The trial friction force is checked against the Coulomb limit, see Eq.
    :eq:`contact-friction-state`.

    ..  math::
        :label: contact-friction-state

        \boldsymbol{f}_t^{trial} &= \lambda_t\ \boldsymbol{e}

        f_t^{limit} &= \mu\ f_n

        \text{state} &= \begin{cases}
            |\boldsymbol{f}_t^{trial}| \leq f_t^{limit} & \text{stick} \\
            \text{else} & \text{slip}
        \end{cases}

    With an allowable elastic slip :math:`\gamma` (``elastic_slip``), the tangential
    multiplier is not constant but proportional to the normal force, see Eq.
    :eq:`contact-friction-elastic-slip`. Hence, a point sticks for an elastic slip
    :math:`|\boldsymbol{e}| \leq \gamma`, independent of the contact pressure, similar
    to the penalty friction of commercial codes.

    ..  math::
        :label: contact-friction-elastic-slip

        \lambda_t = \frac{\mu\ f_n}{\gamma}, \quad
            \frac{\partial \lambda_t}{\partial \Delta \boldsymbol{x}}
            = -\frac{\mu\ \lambda_n}{\gamma}\ \boldsymbol{n}

    In case of sticking contact, the tangential forces are equal to the trial forces,
    see Eq. :eq:`contact-friction-stick`. With an allowable elastic slip, the
    derivative of the tangential multiplier is added to the stiffness matrix,
    :math:`-\mu\ \lambda_n / \gamma\ \boldsymbol{e} \otimes \boldsymbol{n}`.

    ..  math::
        :label: contact-friction-stick

        \boldsymbol{f}_t &= \boldsymbol{f}_t^{trial}

        \boldsymbol{K}_t &= \lambda_t\ \boldsymbol{E}

    In case of sliding contact, the tangential forces are projected to the friction
    limit and the reference gap vectors are updated, see Eq.
    :eq:`contact-friction-slide`.

    ..  math::
        :label: contact-friction-slide

        \hat{\boldsymbol{t}} &= \frac{\boldsymbol{e}}{|\boldsymbol{e}|}

        \boldsymbol{f}_t &= \mu\ f_n\ \hat{\boldsymbol{t}}

        \Delta \boldsymbol{x}^{ref} &=
            \Delta \boldsymbol{x} - \frac{\boldsymbol{f}_t}{\lambda_t}

    The consistent tangential stiffness matrix for sliding contact is given in Eq.
    :eq:`contact-friction-slide-stiffness`. The second term couples the tangential
    forces to the normal force and is not symmetric.

    ..  math::
        :label: contact-friction-slide-stiffness

        \boldsymbol{K}_t &= \frac{\mu\ f_n}{|\boldsymbol{e}|} \left(
            \boldsymbol{1} - \hat{\boldsymbol{t}} \otimes \hat{\boldsymbol{t}}
            \right) \boldsymbol{E} - \mu\ \lambda_n\ \hat{\boldsymbol{t}} \otimes
            \boldsymbol{n}

    With ``symmetric=True``, the symmetric part of the stiffness matrix is returned.

    If ``items`` are given, the multipliers are scaled by the mean normal and the mean
    tangential stiffness of the diagonal blocks of the stiffness matrices of the items,
    evaluated at all points in potential contact, see Eq.
    :eq:`contact-multipliers`. Hence, the multipliers do not depend on the number of
    active contact points. With the default ``multiplier_update="increment"``, the
    multipliers are evaluated on the first evaluation within an increment and are kept
    constant during the Newton-iterations.

    ..  math::
        :label: contact-multipliers

        \lambda_n &= \lambda_n^{\star}\ \frac{1}{n} \sum_a \boldsymbol{K}_{aa} :
            \boldsymbol{P}_n

        \lambda_t &= \lambda_t^{\star}\ \frac{1}{n\ (d-1)} \sum_a \boldsymbol{K}_{aa}
            : \boldsymbol{P}_t

    Examples
    --------
    This example shows how to use a :class:`~felupe.ContactRigidPlane`. An additional
    center-point is added to a mesh.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> mesh.add_points([2.0, 0.5, 0.5])
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
        multiplier=1.0,
        multiplier_tangential=0.1,
        items=None,
        multiplier_update="increment",
        symmetric=False,
        elastic_slip=None,
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.array(points)
        self.centerpoint = centerpoint

        if len(self.points) == 0:
            self.points = self.points.astype(int)

        if self.points.dtype == bool:
            self.points = np.where(self.points)[0]

        if self.centerpoint < 0:
            self.centerpoint = self.mesh.npoints + self.centerpoint

        if self.centerpoint in self.mesh.points_without_cells:
            self.mesh.points_without_cells = self.mesh.points_without_cells[
                self.mesh.points_without_cells != self.centerpoint
            ]

        self.normal = np.array(normal, dtype=float)[: self.mesh.dim]
        self.normal /= np.linalg.norm(self.normal)

        # normal and tangential projections
        self.projection_normal = np.outer(self.normal, self.normal)
        self.projection_tangential = np.eye(self.mesh.dim) - self.projection_normal

        self.friction = friction
        self.items = items

        updates = ["increment", "iteration", "once"]
        if multiplier_update not in updates:
            raise ValueError(f"The multiplier update must be one of {updates}.")

        self.multiplier = multiplier
        self.multiplier_tangential = multiplier_tangential
        self.multiplier_update = multiplier_update
        self.multipliers = None
        self.symmetric = symmetric

        if elastic_slip is not None and not elastic_slip > 0:
            raise ValueError("The elastic slip must be positive.")

        self.elastic_slip = elastic_slip

        # the converged state variables, for which the multipliers were evaluated
        self._multipliers_statevars = None

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.results = Results(stress=False, elasticity=False)

        # differences of undeformed coordinates between points in potential contact
        # and center-point (initial gap vectors)
        self.dX = self.mesh.points[self.points] - self.mesh.points[self.centerpoint]

        # initial state variables: gap vectors, reference (stick) gap vectors and
        # contact flags of the last converged increment
        dx = self._gap_vectors()
        active = dx @ self.normal < 0.0

        self.results.statevars = {"dx": dx, "dx_ref": dx.copy(), "active": active}
        self.results._statevars = None

        # trial results of the last evaluation
        self.results.dx_ref = dx.copy()
        self.results.active = active.copy()
        self.results.slip = np.zeros(len(self.points), dtype=bool)

    def _gap_vectors(self):
        "Return the gap vectors between the points and the center-point."

        u = self.field[0].values

        return self.dX + (u[self.points] - u[self.centerpoint])

    def checkpoint(self):
        """Return a checkpoint of the state of the contact.

        Returns
        -------
        dict
            A dict with copies of the state variables of the last converged increment
            (the gap vectors, the reference gap vectors of the tangential (frictional)
            contact and the mask of active contact points) and the multipliers.

        Notes
        -----
        The state variables are only updated after a converged increment. The field is
        not included in the checkpoint, see e.g.
        :meth:`SolidBody.checkpoint() <felupe.SolidBody.checkpoint>`.

        See Also
        --------
        felupe.ContactRigidPlane.restore : Restore a checkpoint of the contact.
        felupe.CutbackPlugin : A cutback of the increment of failed substeps.
        """

        statevars = {k: v.copy() for k, v in self.results.statevars.items()}

        return {"results.statevars": statevars, "multipliers": self.multipliers}

    def restore(self, checkpoint):
        """Restore a checkpoint of the state of the contact inplace.

        Parameters
        ----------
        checkpoint : dict
            A dict with checkpoint arrays.

        See Also
        --------
        felupe.ContactRigidPlane.checkpoint : Return a checkpoint of the contact.
        """

        statevars = {k: v.copy() for k, v in checkpoint["results.statevars"].items()}

        self.results.statevars = statevars
        self.results._statevars = None

        self.results.dx_ref = statevars["dx_ref"].copy()
        self.results.active = statevars["active"].copy()
        self.results.slip = np.zeros(len(self.points), dtype=bool)

        # multipliers which are updated per increment are re-evaluated
        self.multipliers = checkpoint["multipliers"]
        self._multipliers_statevars = None

        # reset force and stiffness
        self.results.force = None
        self.results.stiffness = None

    def contact_multipliers(self):
        """Return the normal and tangential multipliers.

        Returns
        -------
        ContactMultipliers
            The normal and the tangential multipliers. If items are given, the scale
            factors are multiplied by the mean normal and the mean tangential stiffness
            of the diagonal blocks of the stiffness matrices of the items, evaluated at
            all points in potential contact.
        """

        base_multiplier = 1.0
        base_multiplier_tangential = 1.0

        npoints = len(self.points)

        if self.items is not None and npoints > 0:
            dim = self.mesh.dim
            dof = self.field[0].indices.dof[self.points]

            shape = (npoints, dim, dim)
            rows = np.broadcast_to(dof[:, :, None], shape).ravel()
            cols = np.broadcast_to(dof[:, None, :], shape).ravel()

            blocks = np.zeros(shape)

            for item in self.items:
                if item.results.stiffness is None:
                    _ = item.assemble.matrix(self.field)

                values = item.results.stiffness[rows, cols]
                blocks += np.asarray(values).reshape(shape)

            stiffness = blocks.sum(axis=0)
            base_multiplier = np.sum(stiffness * self.projection_normal) / npoints
            base_multiplier_tangential = np.sum(
                stiffness * self.projection_tangential
            ) / (npoints * max(dim - 1, 1))

        return ContactMultipliers(
            self.multiplier * base_multiplier,
            self.multiplier_tangential * base_multiplier_tangential,
        )

    def _update_multipliers(self):
        "Evaluate the multipliers, depending on the multiplier update."

        statevars = self.results.statevars

        update = self.multipliers is None or self.multiplier_update == "iteration"

        if self.multiplier_update == "increment":
            update = update or (self._multipliers_statevars is not statevars)

        if update:
            self.multipliers = self.contact_multipliers()
            self._multipliers_statevars = statevars

        return self.multipliers

    def _evaluate(self, field=None):
        """Evaluate the contact forces and the stiffness blocks of the points in
        contact for the trial state, based on the converged state variables."""

        if field is not None:
            self.field = field

        dim = self.mesh.dim
        n = self.normal
        P_t = self.projection_tangential
        statevars = self.results.statevars

        dx = self._gap_vectors()
        dtype = dx.dtype

        gap = dx @ n
        active = gap < 0.0
        contact = np.where(active)[0]
        m = len(contact)

        # trial reference gap vectors and sliding flags
        dx_ref = dx.copy()
        slip = np.zeros(len(self.points), dtype=bool)

        force = np.zeros((m, dim), dtype=dtype)
        stiffness = np.zeros((m, dim, dim), dtype=dtype)

        # multipliers (evaluated on the first evaluation within an increment)
        multipliers = self._update_multipliers()
        lambda_n = multipliers.normal
        lambda_t = multipliers.tangential

        if m > 0:
            g = gap[contact]

            # normal contribution
            force += lambda_n * g.reshape(-1, 1) * n
            stiffness += lambda_n * self.projection_normal

            if self.friction > 0.0:
                dx_c = dx[contact]

                # trial elastic slip and its derivative w.r.t. the gap vector
                e = (dx_c - statevars["dx_ref"][contact]) @ P_t
                E = np.tile(P_t, (m, 1, 1)).astype(dtype)

                # points which come into contact within the increment: the reference
                # is the interpolated point of first contact (zero gap)
                new = ~statevars["active"][contact]

                if np.any(new):
                    dx_n = statevars["dx"][contact][new]
                    g_n = dx_n @ n
                    g_new = g[new]
                    delta = (dx_c[new] - dx_n) @ P_t

                    s = np.ones_like(g_new)
                    ds = np.zeros_like(g_new)

                    first = g_n > 0.0
                    s[first] = -g_new[first] / (g_n[first] - g_new[first])
                    ds[first] = -g_n[first] / (g_n[first] - g_new[first]) ** 2

                    e[new] = s.reshape(-1, 1) * delta
                    E[new] = s.reshape(-1, 1, 1) * P_t + ds.reshape(
                        -1, 1, 1
                    ) * np.einsum("ai,j->aij", delta, n)

                force_n_abs = -lambda_n * g
                force_t_limit = self.friction * force_n_abs

                # tangential multipliers: constant or proportional to the normal force
                if self.elastic_slip is None:
                    lambda_t_points = np.full(m, lambda_t, dtype=dtype)
                else:
                    mu = self.friction if np.isfinite(self.friction) else 1.0
                    lambda_t_points = mu * force_n_abs / self.elastic_slip

                force_t_trial = lambda_t_points.reshape(-1, 1) * e
                force_t_norm = np.linalg.norm(force_t_trial, axis=1)

                # points with friction: sticking or sliding
                stick = np.ones(m, dtype=bool)
                if np.isfinite(self.friction):
                    tol = np.sqrt(np.finfo(float).eps) * np.maximum(force_t_limit, 1.0)
                    stick = force_t_norm <= (force_t_limit + tol)

                slide = (~stick) & (force_t_norm > 0.0)

                force_t = force_t_trial
                stiffness_t = lambda_t_points.reshape(-1, 1, 1) * E

                # elastic slip of the (trial) reference gap vectors
                e_ref = e

                if self.elastic_slip is not None:
                    # derivative of the tangential multipliers w.r.t. the gap vector
                    stiffness_t -= (mu * lambda_n / self.elastic_slip) * np.einsum(
                        "ai,j->aij", e, n
                    )

                if np.any(slide):
                    t_hat = force_t_trial[slide] / force_t_norm[slide].reshape(-1, 1)
                    e_norm = np.linalg.norm(e[slide], axis=1)

                    force_t = force_t_trial.copy()
                    force_t[slide] = force_t_limit[slide].reshape(-1, 1) * t_hat

                    # return mapping of the elastic slip
                    e_ref = e.copy()
                    e_ref[slide] *= (
                        force_t_limit[slide] / force_t_norm[slide]
                    ).reshape(-1, 1)

                    # consistent stiffness of the return mapping
                    projection_slip = (
                        np.eye(dim) - np.einsum("ai,aj->aij", t_hat, t_hat)
                    ) / e_norm.reshape(-1, 1, 1)

                    stiffness_t[slide] = self.friction * (
                        force_n_abs[slide].reshape(-1, 1, 1)
                        * projection_slip
                        @ E[slide]
                        - lambda_n * np.einsum("ai,j->aij", t_hat, n)
                    )

                force += force_t
                stiffness += stiffness_t

                slip[contact] = slide

                # updated (trial) reference gap vectors
                dx_ref[contact] = dx_c - e_ref

        if self.symmetric:
            stiffness = (stiffness + stiffness.transpose(0, 2, 1)) / 2

        # trial state variables, these are taken after a converged increment
        self.results._statevars = {"dx": dx, "dx_ref": dx_ref, "active": active}

        self.results.dx_ref = dx_ref
        self.results.active = active
        self.results.slip = slip

        return ContactTrial(contact, force, stiffness, dx, dx_ref, active, slip)

    def _vector(self, field=None, parallel=False):

        trial = self._evaluate(field)
        dof = self.field[0].indices.dof

        self.results.force = _assemble_vector(
            dof=dof[self.points[trial.contact]],
            dof_center=dof[self.centerpoint],
            forces=trial.force,
            size=self.mesh.npoints * self.mesh.dim,
            dtype=trial.dx.dtype,
        )

        return self.results.force

    def _matrix(self, field=None, parallel=False):

        trial = self._evaluate(field)
        dof = self.field[0].indices.dof

        self.results.stiffness = _assemble_matrix(
            dof=dof[self.points[trial.contact]],
            dof_center=dof[self.centerpoint],
            blocks=trial.stiffness,
            size=self.mesh.npoints * self.mesh.dim,
            dtype=trial.dx.dtype,
        )

        return self.results.stiffness

# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from ._base import ContactRigidBase


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


class ContactRigidPlane(ContactPlane, ContactRigidBase):
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
        dim = field.region.mesh.dim

        self.normal = np.array(normal, dtype=float)[:dim]
        self.normal /= np.linalg.norm(self.normal)

        # normal and tangential projections
        self.projection_normal = np.outer(self.normal, self.normal)
        self.projection_tangential = np.eye(dim) - self.projection_normal

        super().__init__(
            field=field,
            points=points,
            centerpoint=centerpoint,
            friction=friction,
            multiplier=multiplier,
            multiplier_tangential=multiplier_tangential,
            items=items,
            multiplier_update=multiplier_update,
            symmetric=symmetric,
            elastic_slip=elastic_slip,
        )

    def _kinematics(self, dx):
        "Return the gaps, the (constant) outward normal vectors and no curvatures."

        gap = dx @ self.normal
        normal = np.broadcast_to(self.normal, dx.shape)

        return gap, normal, None

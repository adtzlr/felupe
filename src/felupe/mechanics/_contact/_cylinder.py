# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from ...view._backends import backend_of
from ...view._shapes import Cylinder, clip_lines, symmetry_clip
from ._base import ContactRigidBase


class ContactRigidCylinder(ContactRigidBase):
    r"""A node-to-surface contact, where the surface is given by a rigid cylinder (a
    rigid circle for 2d-meshes).

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray
        An array with indices of points to be connected to the center-point.
    centerpoint : int
        The index of the centerpoint, located on the axis of the cylinder.
    radius : float
        The radius of the cylinder.
    axis : ndarray or None, optional
        The direction of the axis of the cylinder. Required for 3d-meshes. For
        2d-meshes, the axis is the out-of-plane direction, i.e. the cylinder is a
        circle. Default is None.
    inside : bool, optional
        If False, the points are located outside of the cylinder, e.g. for a
        cylindrical indenter or a roller. If True, the points are located inside of the
        cylinder, e.g. for a rigid tube. Default is False.
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
        :math:`\lambda_t = \mu\ f_n / \gamma`. If None, the tangential multiplier is
        constant. Default is None. See :class:`~felupe.ContactRigidPlane` for details.
    curvature : str or None, optional
        The gaps for the curvature term of the normal stiffness, either
        ``"converged"`` (the gaps of the last converged increment, zero for points
        which come into contact within the increment), ``"trial"`` (the gaps of the
        trial state, i.e. a consistent linearization) or None (no curvature term).
        Default is ``"converged"``.

    Notes
    -----
    A :class:`~felupe.ContactRigidCylinder` is supported as an item in a
    :class:`~felupe.Step`. The contact formulation is the same as for a
    :class:`~felupe.ContactRigidPlane` (penalty-based normal contact, Coulomb friction
    with a return mapping from the converged state, stiffness-based multipliers), but
    the gaps and the outward normal vectors depend on the positions of the points.
    With the projection onto the cross-section of the cylinder, the gaps, the normal
    vectors and their derivatives w.r.t. the gap vectors (the curvatures) are given in
    Eq. :eq:`cylinder-gap`, where :math:`s=1` for points outside and :math:`s=-1` for
    points inside of the cylinder.

    ..  math::
        :label: cylinder-gap

        \boldsymbol{P}_a &= \boldsymbol{1} - \boldsymbol{a} \otimes \boldsymbol{a},
            \quad \boldsymbol{r} = \boldsymbol{P}_a\ \Delta \boldsymbol{x},
            \quad d = |\boldsymbol{r}|

        g &= s\ (d - R), \quad \boldsymbol{n} = s\ \frac{\boldsymbol{r}}{d}

        \boldsymbol{H} &= \frac{\partial \boldsymbol{n}}{\partial \Delta \boldsymbol{x}}
            = \frac{s}{d} \left( \boldsymbol{P}_a - \boldsymbol{n} \otimes
            \boldsymbol{n} \right)

    The consistent normal stiffness includes the curvature term, see Eq.
    :eq:`cylinder-normal-stiffness`. For points outside of the cylinder, this term is
    negative in circumferential direction (the normal force divided by the radius).

    ..  math::
        :label: cylinder-normal-stiffness

        \boldsymbol{f}_n &= \lambda_n\ g\ \boldsymbol{n}

        \boldsymbol{K}_n &= \lambda_n \left( \boldsymbol{n} \otimes \boldsymbol{n}
            + g\ \boldsymbol{H} \right)

    For large increments, the gaps of the trial state in the first Newton-iterations
    may be much larger than the converged gaps. Then, the negative curvature term may
    lead to an indefinite stiffness matrix and the Newton-Raphson method diverges, e.g.
    for soft bodies with free side faces. Hence, the curvature term is evaluated with
    the gaps of the last converged increment by default (``curvature="converged"``),
    i.e. with the normal forces of the last converged increment. This is not a
    consistent linearization, but the curvature term is bounded by the converged
    normal forces. The consistent linearization is used with ``curvature="trial"``.

    The elastic slip is measured in the current tangent plane. Its derivative includes
    the derivative of the tangential projection, see Eq. :eq:`cylinder-friction`. The
    reference gap vectors are updated by the return mapping after each converged
    increment, i.e. the elastic slip is bounded and a large sliding path along the
    circumference is captured by the updates of the reference gap vectors.

    ..  math::
        :label: cylinder-friction

        \boldsymbol{w} &= \Delta \boldsymbol{x} - \Delta \boldsymbol{x}^{ref}_n,
            \quad \boldsymbol{e} = \boldsymbol{P}_t\ \boldsymbol{w}

        \boldsymbol{E} &= \frac{\partial \boldsymbol{e}}{\partial \Delta \boldsymbol{x}}
            = \boldsymbol{P}_t - \left( \boldsymbol{n} \cdot \boldsymbol{w} \right)
            \boldsymbol{H} - \boldsymbol{n} \otimes \boldsymbol{H}\ \boldsymbol{w}

    The cylinder does not rotate, i.e. only the translational degrees of freedom of the
    center-point are coupled.

    Examples
    --------
    This example shows how to use a :class:`~felupe.ContactRigidCylinder`. A rubber
    block is indented by a rigid cylinder, which is dragged along the top face
    afterwards. An additional center-point on the axis of the cylinder is added to the
    mesh.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(b=(3, 1, 1), n=(21, 3, 7))
        >>> mesh.add_points([0.8, 0.5, 1.5])
        >>>
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>>
        >>> umat = fem.NeoHooke(mu=1.0, bulk=5.0)
        >>> solid = fem.SolidBody(umat=umat, field=field)

    The :class:`~felupe.ContactRigidCylinder` connects the center-point with the points
    of the top face, if they are in contact. The axis of the cylinder is parallel to
    the :math:`y`-axis. The allowable elastic slip is about 3 % of the element length.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> contact = fem.ContactRigidCylinder(
        ...     field=field,
        ...     points=np.arange(mesh.npoints)[mesh.z == 1],
        ...     centerpoint=-1,
        ...     radius=0.5,
        ...     axis=[0, 1, 0],
        ...     friction=0.3,
        ...     multiplier=10.0,
        ...     elastic_slip=5e-3,
        ... )
        >>>
        >>> mesh.plot(plotter=contact.plot(size=1.2)).show()

    The mesh is fixed on the bottom face. The center-point is moved downwards first and
    horizontally afterwards. For large increments, the contact status of the points
    and the stick-slip states may alternate between the Newton-iterations. Then, a
    :class:`~felupe.CutbackPlugin` reduces the increment of failed substeps.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {
        ...     "fixed": fem.Boundary(displacement, fz=0),
        ...     "center": fem.Boundary(displacement, fz=1.5, skip=(1, 0, 1)),
        ...     "move-z": fem.Boundary(displacement, fz=1.5, skip=(1, 1, 0)),
        ...     "move-x": fem.Boundary(displacement, fz=1.5, skip=(0, 1, 1)),
        ... }
        >>> step = fem.Step(
        ...     [solid, contact],
        ...     boundaries=boundaries,
        ...     ramp={
        ...         boundaries["move-z"]: fem.math.linsteps([0, -0.15, -0.15], num=[5, 20]),
        ...         boundaries["move-x"]: fem.math.linsteps([0, 0, 0.5], num=[5, 20]),
        ...     },
        ... )
        >>> job = fem.Job([step]).evaluate()

    A view on the deformed mesh including the :class:`~felupe.ContactRigidCylinder` is
    plotted.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> field.plot(
        ...     "Displacement",
        ...     component=0,
        ...     show_undeformed=False,
        ...     plotter=contact.plot(size=1.2),
        ...     clim=[0, 0.08],
        ... ).show()

    See Also
    --------
    felupe.ContactRigidPlane : A node-to-surface contact, where the surface is given by
        a rigid plane.
    """

    def __init__(
        self,
        field,
        points,
        centerpoint,
        radius,
        axis=None,
        inside=False,
        friction=0.0,
        multiplier=1.0,
        multiplier_tangential=0.1,
        items=None,
        multiplier_update="increment",
        symmetric=False,
        elastic_slip=None,
        curvature="converged",
    ):
        dim = field.region.mesh.dim

        if not radius > 0:
            raise ValueError("The radius must be positive.")

        self.radius = float(radius)
        self.inside = bool(inside)
        self.sign = -1.0 if self.inside else 1.0

        if axis is None:
            if dim == 3:
                raise ValueError("The axis of the cylinder is required for 3d-meshes.")
            axis = [0.0, 0.0, 1.0]

        self.axis = np.pad(np.array(axis, dtype=float).ravel(), (0, 3))[:3]
        norm = np.linalg.norm(self.axis)

        if not norm > 0:
            raise ValueError("The axis of the cylinder must not be a zero vector.")

        self.axis /= norm

        if dim == 2 and not np.isclose(abs(self.axis[2]), 1.0):
            raise ValueError("The axis must be the out-of-plane direction in 2d.")

        # projection onto the cross-section of the cylinder
        a = self.axis[:dim]
        self.projection_axial = np.eye(dim) - np.outer(a, a)

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
            curvature=curvature,
        )

    def _kinematics(self, dx):
        "Return the gaps, the outward normal vectors and the curvatures."

        r = dx @ self.projection_axial
        d = np.linalg.norm(r, axis=1)
        d = np.maximum(d, np.sqrt(np.finfo(float).eps) * self.radius)

        n0 = r / d.reshape(-1, 1)

        gap = self.sign * (d - self.radius)
        normal = self.sign * n0
        curvature = (
            self.sign
            * (self.projection_axial - np.einsum("ai,aj->aij", n0, n0))
            / d.reshape(-1, 1, 1)
        )

        return gap, normal, curvature

    def plot(
        self,
        plotter=None,
        deformed=True,
        size=None,
        resolution=120,
        show_edges=False,
        color="black",
        opacity=0.5,
        line_width=None,
        show_point=True,
        show_line=True,
        sym=(False, False, False),
        **kwargs,
    ):
        """Plot the rigid cylinder (a circle for 2d-meshes) with its center-point.

        Parameters
        ----------
        plotter : pyvista.Plotter or None, optional
            An existing plotter. Default is None.
        deformed : bool, optional
            A flag to plot the cylinder at the deformed center-point. Default is True.
        size : float or None, optional
            The length of the cylinder (3d only). If None, the length is 1.05 times the
            largest dimension of the mesh. Default is None.
        resolution : int, optional
            The number of points in circumferential direction. Default is 120.
        show_edges : bool, optional
            A flag to show the edges of the cylinder surface (3d only). Default is
            False.
        color : str, optional
            The color of the cylinder. Default is "black".
        opacity : float, optional
            The opacity of the cylinder. Default is 0.5.
        line_width : float or None, optional
            The line width of the circle (2d only) and the axis. Default is None.
        show_point : bool, optional
            A flag to show the center-point. Default is True.
        show_line : bool, optional
            A flag to show the axis of the cylinder (3d only). Default is True.
        sym : tuple of bool, optional
            Clip the cylinder at the symmetry planes of the given axes. Default is
            (False, False, False).
        **kwargs : dict, optional
            Additional keyword arguments for the meshes of the plotter.

        Returns
        -------
        pyvista.Plotter
            The plotter with the cylinder.
        """

        backend = backend_of(plotter)

        if plotter is None:
            plotter = backend.prepare_plotter(theme=None)

        dim = self.mesh.dim
        x = self.mesh.points

        if deformed:
            x = self.mesh.points + self.field[0].values

        center = np.pad(x[self.centerpoint], (0, 3 - dim))

        if size is None:
            size = 1.05 * (x.max(axis=0) - x.min(axis=0)).max()

        clip = symmetry_clip(sym, self.mesh.points)
        options = dict(color=color, opacity=opacity, show_edges=show_edges, **kwargs)

        if dim == 2:
            # the circle as closed polyline of line segments
            phi = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
            circle = np.zeros((resolution, 3))
            circle[:, 0] = self.radius * np.cos(phi)
            circle[:, 1] = self.radius * np.sin(phi)
            points = center + circle
            lines = clip_lines(np.stack([points, np.roll(points, -1, axis=0)], 1), clip)

            if len(lines) > 0:
                plotter = backend.add_lines(
                    plotter, lines, line_width=line_width, **options
                )

        else:
            cylinder = Cylinder(
                center=center,
                direction=self.axis,
                radius=self.radius,
                height=size,
                resolution=resolution,
                capping=False,
                clip=clip,
            )
            plotter = backend.add_shape(
                plotter, cylinder, line_width=line_width, **options
            )

        if show_point:
            plotter = backend.add_points(
                plotter,
                center.reshape(1, -1),
                color=color,
                point_size=12,
                opacity=opacity,
                **kwargs,
            )

        if show_line and dim == 3:
            axis = [
                [center - 0.55 * size * self.axis, center + 0.55 * size * self.axis]
            ]
            plotter = backend.add_lines(
                plotter,
                axis,
                color=color,
                line_width=line_width,
                opacity=opacity,
                **kwargs,
            )

        return plotter

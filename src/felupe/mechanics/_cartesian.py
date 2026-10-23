# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from ..view._backends import backend_of
from ..view._shapes import Cone
from ._pointconstraint import PointConstraint


class CartesianConstraint(PointConstraint):
    r"""A constraint for the displacements of points in a (fixed) Cartesian coordinate
    system.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray of int or bool
        An array with indices (or a mask) of the points to be constrained.
    rotation : array_like or None, optional
        The rotation matrix :math:`\boldsymbol{Q} = \boldsymbol{e}_i \otimes
        \boldsymbol{E}_i`, which maps the global base vectors :math:`\boldsymbol{E}_i`
        to the base vectors :math:`\boldsymbol{e}_i = \boldsymbol{Q}
        \boldsymbol{E}_i` of the coordinate system, with shape (3, 3) or with shape
        (2, 2) for 2d-meshes, see :func:`~felupe.math.rotation_matrix`. The base vectors
        are the columns of the matrix, i.e. given base vectors are stacked by
        ``np.column_stack([e1, e2, e3])``. The columns are normalized, they must be
        orthogonal and right-handed (:math:`\det(\boldsymbol{Q}) = 1`). If None, the
        global base vectors are used. Default is None.
    displacement : array_like, optional
        The prescribed displacements :math:`(u_1, u_2, u_3)` in the directions of the
        base vectors for all points or per point with shape (n, 3). May be ramped in a
        :class:`~felupe.Step`, see :meth:`update`. Default is (0, 0, 0).
    skip : tuple of bool, optional
        Flags to skip the directions of the base vectors. Default is
        (False, False, False).
    multiplier : float, optional
        The penalty stiffness :math:`k` per point and direction. Default is 1e3.

    Attributes
    ----------
    rotation : (3, 3) ndarray
        The rotation matrix with the normalized base vectors of the coordinate system
        as columns (padded to three dimensions for 2d-meshes).
    results : Results
        The results of the last evaluation, including the gaps ``results.gap`` with
        shape (n, 3).

    Notes
    -----
    A :class:`~felupe.CartesianConstraint` is supported as an item in a
    :class:`~felupe.Step`. The coordinate system is fixed in space, i.e. its base
    vectors :math:`\boldsymbol{e}_i` do not rotate with the deformation. The
    constraint is enforced by a penalty method with the potential in
    Eq. :eq:`cartesian-potential`, summed up for all points and all directions which
    are not skipped.

    ..  math::
        :label: cartesian-potential

        \Pi = \sum_{\text{points}} \sum_{i \in \{1, 2, 3\}} \frac{k}{2}\, g_i^2, \qquad
        g_i = \boldsymbol{e}_i \cdot \boldsymbol{u} - u_i

    The gaps are linear in the displacements :math:`\boldsymbol{u}` of a point, hence,
    the residual vector and the (constant) stiffness matrix of a point are given in
    Eq. :eq:`cartesian-vector-matrix`.

    ..  math::
        :label: cartesian-vector-matrix

        \boldsymbol{r} = \sum_i k\, g_i\, \boldsymbol{e}_i, \qquad
        \boldsymbol{K} = \sum_i k\, \boldsymbol{e}_i \otimes \boldsymbol{e}_i

    If all directions are constrained, the stiffness matrix is :math:`k \boldsymbol{1}`
    and the coordinate system only transforms the prescribed displacements. Then, a
    :class:`~felupe.Boundary` with the values
    :math:`\boldsymbol{u} = \sum_i u_i\, \boldsymbol{e}_i` is exact and should be
    preferred. A Cartesian constraint is useful for skipped directions, e.g. for
    supports on inclined planes, where the points slide in a plane (or along a line)
    which is not aligned to the global axes.

    Examples
    --------
    A cube is rotated by 30° around the :math:`y`-axis. The base vectors of the
    coordinate system are the global base vectors rotated like the mesh, i.e. the
    columns of the rotation matrix.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6).rotate(30, axis=1)
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
        >>>
        >>> rotation = fem.math.rotation_matrix(30, dim=3, axis=1)
        >>> local = mesh.points @ rotation

    The bottom face rests on an inclined plane, i.e. it is constrained in the normal
    direction of the plane only and the points slide on the plane. The top face is
    moved down in global :math:`z`-direction by a plate, which is fixed in :math:`x`-
    and :math:`y`-direction.

    ..  pyvista-plot::
        :context:

        >>> support = fem.CartesianConstraint(
        ...     field,
        ...     points=np.isclose(local[:, 2], 0),
        ...     rotation=rotation,
        ...     skip=(1, 1, 0),
        ... )
        >>> top = np.isclose(local[:, 2], 1)
        >>> boundaries = {
        ...     "plate": fem.Boundary(displacement, mask=top, skip=(0, 0, 1)),
        ...     "move": fem.Boundary(displacement, mask=top, skip=(1, 1, 0)),
        ... }
        >>> move = fem.math.linsteps([0, -0.3], num=5)
        >>> step = fem.Step(
        ...     [solid, support], ramp={boundaries["move"]: move}, boundaries=boundaries
        ... )
        >>> job = fem.Job([step]).evaluate()

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> plotter = support.plot(color="red")
        >>> solid.plot("Principal Values of Cauchy Stress", plotter=plotter).show()

    See Also
    --------
    felupe.CylindricalConstraint : A constraint for the displacements of points in
        cylindrical coordinates.
    felupe.Boundary : A collection of prescribed degrees of freedom.

    """

    def __init__(
        self,
        field,
        points,
        rotation=None,
        displacement=(0.0, 0.0, 0.0),
        skip=(False, False, False),
        multiplier=1e3,
    ):
        super().__init__(field, points, displacement, skip, multiplier)

        if rotation is None:
            rotation = np.eye(self.dim)

        rotation = np.asarray(rotation, dtype=float)

        if rotation.shape != (self.dim, self.dim):
            raise ValueError(
                f"The rotation must be given with shape ({self.dim}, {self.dim}), "
                f"got {rotation.shape}."
            )

        norm = np.linalg.norm(rotation, axis=0)

        if np.any(np.isclose(norm, 0)):
            raise ValueError("The base vectors must not be zero-vectors.")

        rotation = rotation / norm

        if not np.allclose(rotation.T @ rotation, np.eye(self.dim)):
            raise ValueError("The base vectors must be orthogonal.")

        if np.linalg.det(rotation) < 0:
            raise ValueError(
                "The base vectors must be right-handed, i.e. det(rotation) = 1."
            )

        self.rotation = np.eye(3)
        self.rotation[: self.dim, : self.dim] = rotation

    def _gaps(self, u, stiffness=True):
        "Evaluate the (linear) gaps and their gradients."

        gap = u @ self.rotation - self.displacement
        dgdx = np.broadcast_to(self.rotation.T, (len(self.points), 3, 3))

        return gap, dgdx, None

    def _plot_reference(self, plotter, x, color, line_width):
        """Add arrows of the base vectors of the constrained directions, which point to
        the center of the points (like the symbols of supports)."""

        backend = backend_of(plotter)
        center = x.mean(axis=0)
        length = max(0.25 * np.ptp(self._pad(self.mesh.points), axis=0).max(), 1e-12)

        for direction in self.rotation.T[self.mask]:
            start = center - length * direction
            tip = center - 0.2 * length * direction
            cone = Cone(
                center=center - 0.1 * length * direction,
                direction=direction,
                height=0.2 * length,
                radius=0.07 * length,
            )
            backend.add_lines(
                plotter, [[start, tip]], color=color, line_width=line_width
            )
            backend.add_shape(plotter, cone, color=color)

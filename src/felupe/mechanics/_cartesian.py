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
    basis : array_like or None, optional
        The matrix with the base vectors :math:`\boldsymbol{e}_i` of the coordinate
        system as columns, with shape (3, 3) or with shape (2, 2) for 2d-meshes, i.e.
        the rotation :math:`\boldsymbol{Q} = \boldsymbol{e}_i \otimes
        \boldsymbol{E}_i` of the global base vectors :math:`\boldsymbol{E}_i`, see
        :func:`~felupe.math.rotation_matrix`. Given base vectors are stacked as
        columns by ``np.column_stack([e1, e2, e3])``. The base vectors are normalized
        and they must be orthogonal. If None, the global base vectors are used.
        Default is None.
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
    basis : (3, 3) ndarray
        The normalized base vectors of the coordinate system as columns (padded to
        three dimensions for 2d-meshes).
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
        >>> basis = fem.math.rotation_matrix(30, dim=3, axis=1)
        >>> local = mesh.points @ basis

    The bottom face rests on an inclined plane, i.e. it is constrained in the normal
    direction of the plane only and the points slide on the plane. The top face is
    moved down in global :math:`z`-direction by a plate, which is fixed in :math:`x`-
    and :math:`y`-direction.

    ..  pyvista-plot::
        :context:

        >>> support = fem.CartesianConstraint(
        ...     field, points=np.isclose(local[:, 2], 0), basis=basis, skip=(1, 1, 0)
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
        basis=None,
        displacement=(0.0, 0.0, 0.0),
        skip=(False, False, False),
        multiplier=1e3,
    ):
        super().__init__(field, points, displacement, skip, multiplier)

        if basis is None:
            basis = np.eye(self.dim)

        basis = np.asarray(basis, dtype=float)

        if basis.shape != (self.dim, self.dim):
            raise ValueError(
                f"The basis must be given with shape ({self.dim}, {self.dim}), "
                f"got {basis.shape}."
            )

        norm = np.linalg.norm(basis, axis=0)

        if np.any(np.isclose(norm, 0)):
            raise ValueError("The base vectors must not be zero-vectors.")

        basis = basis / norm

        if not np.allclose(basis.T @ basis, np.eye(self.dim)):
            raise ValueError("The base vectors must be orthogonal.")

        self.basis = np.eye(3)
        self.basis[: self.dim, : self.dim] = basis

    def _gaps(self, u, stiffness=True):
        "Evaluate the (linear) gaps and their gradients."

        gap = u @ self.basis - self.displacement
        dgdx = np.broadcast_to(self.basis.T, (len(self.points), 3, 3))

        return gap, dgdx, None

    def _plot_reference(self, plotter, x, color, line_width):
        """Add arrows of the base vectors of the constrained directions, which point to
        the center of the points (like the symbols of supports)."""

        import pyvista as pv

        center = x.mean(axis=0)
        length = max(0.25 * np.ptp(self._pad(self.mesh.points), axis=0).max(), 1e-12)

        for direction in self.basis.T[self.mask]:
            start = center - length * direction
            tip = center - 0.2 * length * direction
            plotter.add_mesh(pv.Line(start, tip), color=color, line_width=line_width)
            plotter.add_mesh(
                pv.Cone(
                    center=center - 0.1 * length * direction,
                    direction=direction,
                    height=0.2 * length,
                    radius=0.07 * length,
                ),
                color=color,
            )

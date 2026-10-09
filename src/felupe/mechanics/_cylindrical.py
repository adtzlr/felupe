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
from scipy.sparse import coo_matrix, csr_matrix

from ._helpers import Assemble, Results


class CylindricalConstraint:
    r"""A constraint for the displacements of points in cylindrical coordinates.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray of int or bool
        An array with indices (or a mask) of the points to be constrained.
    center : array_like, optional
        A point on the axis. Default is (0, 0, 0).
    axis : array_like, optional
        The direction of the axis, perpendicular to the plane for 2d-meshes. Default is
        (0, 0, 1).
    displacement : array_like, optional
        The prescribed displacements :math:`(u_r, u_\theta, u_z)` for all points or per
        point with shape (n, 3), where :math:`u_\theta` is the arc length on the
        undeformed radius. May be ramped in a :class:`~felupe.Step`, see
        :meth:`update`. Default is (0, 0, 0).
    skip : tuple of bool, optional
        Flags to skip the radial, the circumferential and the axial direction. Default
        is (False, False, False).
    multiplier : float, optional
        The penalty stiffness :math:`k` per point and direction. Default is 1e3.

    Attributes
    ----------
    radius : (n,) ndarray
        The undeformed radii :math:`R` of the points.
    results : Results
        The results of the last evaluation, including the gaps ``results.gap`` with
        shape (n, 3).

    Notes
    -----
    A :class:`~felupe.CylindricalConstraint` is supported as an item in a
    :class:`~felupe.Step`. For each point, the undeformed radius, the axial coordinate
    and the cylindrical base vectors are evaluated w.r.t. the center-point
    :math:`\boldsymbol{X}_0` and the normalized axis :math:`\boldsymbol{a}`, see
    Eq. :eq:`cylindrical-basis`.

    ..  math::
        :label: cylindrical-basis

        \boldsymbol{P} &= \boldsymbol{1} - \boldsymbol{a} \otimes \boldsymbol{a}

        R &= \| \boldsymbol{P} (\boldsymbol{X} - \boldsymbol{X}_0) \|, \qquad
        Z = \boldsymbol{a} \cdot (\boldsymbol{X} - \boldsymbol{X}_0)

        \boldsymbol{e}_r &= \frac{1}{R}\, \boldsymbol{P} (\boldsymbol{X} -
        \boldsymbol{X}_0), \qquad
        \boldsymbol{e}_\theta = \boldsymbol{a} \times \boldsymbol{e}_r, \qquad
        \boldsymbol{e}_z = \boldsymbol{a}

    The constraint is enforced by a penalty method with the potential in
    Eq. :eq:`cylindrical-potential`, summed up for all points and all directions which
    are not skipped.

    ..  math::
        :label: cylindrical-potential

        \Pi = \sum_{\text{points}} \sum_{i \in \{r, \theta, z\}} \frac{k}{2}\, g_i^2

    The gaps are evaluated in the deformed configuration
    :math:`\boldsymbol{x} = \boldsymbol{X} + \boldsymbol{u}` with the base vectors
    rotated by the angle :math:`\varphi = u_\theta / R`, see Eq. :eq:`cylindrical-gaps`.
    If the circumferential direction is skipped, the radial gap is the exact distance to
    the axis, i.e. the points slide on the cylinder like in a rigid sleeve.

    ..  math::
        :label: cylindrical-gaps

        \boldsymbol{e}_r(\varphi) &= \cos(\varphi)\, \boldsymbol{e}_r
            + \sin(\varphi)\, \boldsymbol{e}_\theta, \qquad
        \boldsymbol{e}_\theta(\varphi) = -\sin(\varphi)\, \boldsymbol{e}_r
            + \cos(\varphi)\, \boldsymbol{e}_\theta

        g_r &= \begin{cases}
            \boldsymbol{e}_r(\varphi) \cdot (\boldsymbol{x} - \boldsymbol{X}_0)
            - (R + u_r) & \text{circumferential direction constrained} \\
            \| \boldsymbol{P} (\boldsymbol{x} - \boldsymbol{X}_0) \| - (R + u_r)
            & \text{circumferential direction skipped}
        \end{cases}

        g_\theta &= \boldsymbol{e}_\theta(\varphi) \cdot
            (\boldsymbol{x} - \boldsymbol{X}_0)

        g_z &= \boldsymbol{a} \cdot (\boldsymbol{x} - \boldsymbol{X}_0) - (Z + u_z)

    The residual vector and the stiffness matrix of a point are given in
    Eq. :eq:`cylindrical-vector-matrix`. The only non-zero second derivative is the one
    of the exact radial gap with the deformed radius :math:`\rho` and the deformed
    radial direction :math:`\boldsymbol{n}`.

    ..  math::
        :label: cylindrical-vector-matrix

        \boldsymbol{r} &= \sum_i k\, g_i\, \frac{\partial g_i}{\partial \boldsymbol{x}}

        \boldsymbol{K} &= \sum_i k \left(
            \frac{\partial g_i}{\partial \boldsymbol{x}} \otimes
            \frac{\partial g_i}{\partial \boldsymbol{x}}
            + g_i\, \frac{\partial^2 g_i}{
                \partial \boldsymbol{x}\, \partial \boldsymbol{x}
            }
        \right)

        \frac{\partial^2 g_r}{\partial \boldsymbol{x}\, \partial \boldsymbol{x}} &=
        \frac{1}{\rho} \left( \boldsymbol{P} - \boldsymbol{n} \otimes \boldsymbol{n}
        \right)

    For 2d-meshes, the constraint is formulated in polar coordinates. Points on the axis
    can't be constrained in radial or circumferential direction.

    Examples
    --------
    A tube with its axis in direction :math:`x` is created by a revolution of a
    rectangle.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> rectangle = fem.Rectangle(a=(0, 1), b=(3, 2), n=(10, 4))
        >>> mesh = rectangle.revolve(n=25, phi=360, axis=0)
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)

    The outer surface is placed in a rigid sleeve, i.e. it is constrained in radial
    direction only. The end face at :math:`x=0` is fixed and the end face at
    :math:`x=3` is twisted by 90°, where the circumferential displacement is the arc
    length on the undeformed radius of each point.

    ..  pyvista-plot::
        :context:

        >>> radius = np.linalg.norm(mesh.points[:, 1:], axis=1)
        >>> sleeve = fem.CylindricalConstraint(
        ...     field, points=np.isclose(radius, 2), axis=(1, 0, 0), skip=(0, 1, 1)
        ... )
        >>> twist = fem.CylindricalConstraint(
        ...     field, points=np.isclose(mesh.x, 3), axis=(1, 0, 0)
        ... )
        >>> angles = np.linspace(0, np.pi / 2, 10)
        >>> table = [twist.radius.reshape(-1, 1) * [0, angle, 0] for angle in angles]
        >>>
        >>> boundaries = {"fixed": fem.Boundary(displacement, fx=0)}
        >>> step = fem.Step(
        ...     [solid, sleeve, twist], ramp={twist: table}, boundaries=boundaries
        ... )
        >>> job = fem.Job([step]).evaluate()

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> plotter = twist.plot(color="red")
        >>> solid.plot("Principal Values of Cauchy Stress", plotter=plotter).show()

    See Also
    --------
    felupe.MultiPointConstraint : A multi-point constraint which connects a
        center-point to a list of points.
    felupe.Boundary : A collection of prescribed degrees of freedom.

    """

    def __init__(
        self,
        field,
        points,
        center=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        displacement=(0.0, 0.0, 0.0),
        skip=(False, False, False),
        multiplier=1e3,
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.dim = self.mesh.dim
        self.points = np.array(points)
        self.multiplier = multiplier

        if self.dim not in [2, 3]:
            raise ValueError("A cylindrical constraint requires a 2d- or a 3d-mesh.")

        if len(self.points) == 0:
            self.points = self.points.astype(int)

        if self.points.dtype == bool:
            self.points = np.where(self.points)[0]

        self.center = self._pad(np.asarray(center, dtype=float).ravel())

        axis = self._pad(np.asarray(axis, dtype=float).ravel())
        norm = np.linalg.norm(axis)

        if np.isclose(norm, 0):
            raise ValueError("The axis must not be a zero-vector.")

        self.axis = axis / norm

        if self.dim == 2 and not np.isclose(abs(self.axis[2]), 1):
            raise ValueError(
                "The axis must be perpendicular to the plane of a 2d-mesh."
            )

        # mask of the constrained directions (radial, circumferential, axial)
        skip = np.asarray(skip, dtype=bool).ravel()
        skip = np.pad(skip, (0, 3 - len(skip)), constant_values=True)
        self.mask = ~skip

        if self.dim == 2:
            self.mask[2] = False

        # undeformed radii, axial coordinates and cylindrical base vectors
        dX = self._pad(self.mesh.points[self.points]) - self.center
        self.projection = np.eye(3) - np.outer(self.axis, self.axis)

        dXr = dX @ self.projection
        self.radius = np.linalg.norm(dXr, axis=1)
        self.height = dX @ self.axis

        if np.any(self.mask[:2]) and np.any(np.isclose(self.radius, 0)):
            raise ValueError(
                "Points on the axis can't be constrained in radial or circumferential "
                "direction."
            )

        e_r = np.zeros_like(dXr)
        np.divide(
            dXr,
            self.radius.reshape(-1, 1),
            out=e_r,
            where=self.radius.reshape(-1, 1) > 0,
        )
        e_t = np.cross(self.axis, e_r)
        e_z = np.broadcast_to(self.axis, e_r.shape)

        # cylindrical base vectors, shape (n, 3, 3): (point, direction, component)
        self.basis = np.stack([e_r, e_t, e_z], axis=1)

        self.displacement = None
        self.update(displacement)

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def _pad(self, values):
        "Pad the last axis of an array with zeros to three components."
        width = [(0, 0)] * (values.ndim - 1) + [(0, 3 - values.shape[-1])]
        return np.pad(values, width)

    def update(self, displacement):
        r"""Update the prescribed displacements :math:`(u_r, u_\theta, u_z)`.

        Parameters
        ----------
        displacement : array_like
            The prescribed displacements in radial, circumferential and axial
            direction, either one vector for all points or one vector per point with
            shape (n, 3). For 2d-meshes, also vectors with two components
            :math:`(u_r, u_\theta)` are supported.

        Notes
        -----
        This method is called by a :class:`~felupe.Step` for each substep if the
        constraint is ramped, e.g. by ``ramp={constraint: table}``.
        """

        displacement = np.asarray(displacement, dtype=float)

        if displacement.ndim not in [1, 2] or displacement.shape[-1] not in [
            self.dim,
            3,
        ]:
            raise ValueError(
                "The displacements must be given with shape (3,) or (n, 3), "
                f"got {displacement.shape}."
            )

        displacement = self._pad(displacement)
        self.displacement = np.broadcast_to(displacement, (len(self.points), 3)).copy()

    def _evaluate(self, field=None, stiffness=True):
        "Evaluate the gaps, the forces and (optionally) the stiffness of all points."

        if field is not None:
            self.field = field

        u = self._pad(self.field[0].values[self.points])
        e_r, e_t, e_z = self.basis.transpose([1, 0, 2])
        R = self.radius
        u_r, u_t, u_z = self.displacement.T

        npoints = len(self.points)
        gap = np.zeros((npoints, 3))
        force = np.zeros((npoints, 3))
        matrix = np.zeros((npoints, 3, 3)) if stiffness else None

        dot = lambda a, b: np.einsum("ai,ai->a", a, b)
        dya = lambda a, b: np.einsum("ai,aj->aij", a, b)

        def add(i, g, dgdx):
            gap[:, i] = g
            force[:] += g.reshape(-1, 1) * dgdx

            if stiffness:
                matrix[:] += dya(dgdx, dgdx)

        # base vectors rotated by the prescribed angle (ignored if skipped)
        phi = np.zeros((npoints, 1))

        if self.mask[1]:
            np.divide(u_t, R, out=phi[:, 0], where=R > 0)

        cos, sin = np.cos(phi), np.sin(phi)
        e_r, e_t = cos * e_r + sin * e_t, -sin * e_r + cos * e_t

        # projections of the undeformed points on the rotated base vectors, i.e.
        # e_r(phi) . (X - X0) - R = -2 R sin^2(phi / 2) and e_t(phi) . (X - X0)
        dX_r = -2 * R * np.sin(phi.ravel() / 2) ** 2
        dX_t = -R * sin.ravel()

        if self.mask[0]:  # radial direction
            if not self.mask[1]:  # slide on the cylinder
                dXr = R.reshape(-1, 1) * self.basis[:, 0]
                du = u @ self.projection
                dxr = dXr + du
                rho = np.linalg.norm(dxr, axis=1)
                n = dxr / rho.reshape(-1, 1)

                # g = rho - (R + u_r), evaluated without cancellation (|dXr| = R)
                g = (2 * dot(dXr, du) + dot(du, du) - (2 * R + u_r) * u_r) / (
                    rho + R + u_r
                )
                add(0, g, n)

                if stiffness:
                    nn = dya(n, n)
                    matrix[:] += (g / rho).reshape(-1, 1, 1) * (self.projection - nn)

            else:
                add(0, dot(e_r, u) + dX_r - u_r, e_r)

        if self.mask[1]:  # circumferential direction
            add(1, dot(e_t, u) + dX_t, e_t)

        if self.mask[2]:  # axial direction
            add(2, dot(e_z, u) - u_z, e_z)

        self.results.gap = gap

        force = self.multiplier * force[:, : self.dim]

        if stiffness:
            matrix = self.multiplier * matrix[:, : self.dim, : self.dim]

        return force, matrix

    def _vector(self, field=None, parallel=False):
        "Calculate the vector of residuals of the cylindrical constraint."

        force, _ = self._evaluate(field, stiffness=False)

        dof = self.field[0].indices.dof[self.points].ravel()
        ndof = self.field[0].values.size

        self.results.force = csr_matrix(
            (force.ravel(), (dof, np.zeros_like(dof))), shape=(ndof, 1)
        )
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate the stiffness matrix of the cylindrical constraint."

        _, matrix = self._evaluate(field, stiffness=True)

        dof = self.field[0].indices.dof[self.points]
        ndof = self.field[0].values.size

        rows = np.repeat(dof, self.dim, axis=1).ravel()
        cols = np.tile(dof, (1, self.dim)).ravel()

        self.results.stiffness = coo_matrix(
            (matrix.ravel(), (rows, cols)), shape=(ndof, ndof)
        ).tocsr()
        return self.results.stiffness

    def plot(
        self,
        plotter=None,
        color="black",
        deformed=True,
        point_size=10,
        line_width=2,
        **kwargs,
    ):
        """Plot the constrained points and the axis of the cylindrical constraint.

        Parameters
        ----------
        plotter : pyvista.Plotter or None, optional
            An existing plotter. If None, a new plotter is created. Default is None.
        color : str, optional
            The color of the points and the axis. Default is "black".
        deformed : bool, optional
            A flag to plot the points in the deformed configuration. Default is True.
        point_size : float, optional
            The size of the points. Default is 10.
        line_width : float, optional
            The line width of the axis. Default is 2.
        **kwargs : dict, optional
            Additional keyword arguments for the points, passed to
            :meth:`pyvista.Plotter.add_points`.

        Returns
        -------
        pyvista.Plotter
            The plotter with the points and the axis.
        """

        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()

        x = self.mesh.points[self.points]

        if deformed:
            x = x + self.field[0].values[self.points]

        x = self._pad(x)

        if len(x) > 0:
            plotter.add_points(x, color=color, point_size=point_size, **kwargs)

            # axis line along the axial extent of the points (or their mean radius)
            z = (x - self.center) @ self.axis
            length = max(np.ptp(z), self.radius.mean(), 1e-12)
            start = self.center + (z.min() - 0.1 * length) * self.axis
            end = self.center + (z.min() + 1.1 * length) * self.axis
            plotter.add_mesh(pv.Line(start, end), color=color, line_width=line_width)

        return plotter

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


class RigidPlaneContact:
    """A node-to-surface contact, where the surface is given by a rigid plane.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray
        An array with indices of points to be connected to the center-point.
    centerpoint : int
        The index of the centerpoint.
    normal : ndarray
        The plane normal vector.
    stick : bool, optional
        A flag to activate tangential stick. Default is False (frictionless contact).
    multiplier : float, optional
        A multiplier to penalize the relative displacements between the center-point and
        the points in contact. Default is 1e-6.

    Notes
    -----
    A :class:`~felupe.MultiPointContact` is supported as an item in a
    :class:`~felupe.Step`. It provides the assemble-methods
    :meth:`MultiPointContact.assemble.vector() <felupe.MultiPointContact.assemble.vector>`
    and :meth:`MultiPointContact.assemble.matrix() <felupe.MultiPointContact.assemble.matrix>`.

    Examples
    --------
    This example shows how to use a :class:`~felupe.MultiPointContact`.

    An additional center-point is added to a mesh. By default, all *hanging* points are
    collected in the mesh-attribute
    :attr:`Mesh.points_without_cells <felupe.Mesh.points_without_cells>`. The degrees of
    freedom of these points are considered as fixed, i.e. they are ignored. The center-
    point is not connected to any cell and is added to the points-without-cells array
    on :meth:`Mesh.update <felupe.Mesh.update>`. Hence, center-point has to be removed
    manually.

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

    A :class:`~felupe.MultiPointContact` defines the multi-point contact which
    connects the displacement degrees of freedom of the center-point with the dofs of
    points located at :math:`x=1` if they are in contact. Only the :math:`x`-component
    is considered in this example.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> contact = fem.MultiPointContact(
        ...     field=field,
        ...     points=np.arange(mesh.npoints)[mesh.x == 1],
        ...     centerpoint=-1,
        ...     skip=(False, True, True)
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
        >>> mesh.plot(plotter=contact.plot(plotter=plotter)).show()

    The mesh is fixed on the left end face and a ramped :class:`~felupe.Boundary` is
    applied on the center-point of the :class:`~felupe.MultiPointContact`. All items
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

    A view on the deformed mesh including the :class:`~felupe.MultiPointContact` is
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
        >>> field.plot("Displacement", component=None, plotter=contact.plot(plotter=plotter)).show()

    See Also
    --------
    felupe.MultiPointConstraint : A Multi-point-constraint which connects a center-point
        to a list of points.

    """

    def __init__(
        self,
        field,
        points,
        centerpoint,
        normal,
        stick=False,
        multiplier=1e6,
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.array(points)
        self.centerpoint = centerpoint

        self.normal = np.array(normal, dtype=float)[: self.mesh.dim]
        self.normal /= np.linalg.norm(self.normal)

        # normal and tangential projections
        self.Pn = np.outer(self.normal, self.normal)
        self.Pt = np.eye(self.mesh.dim) - self.Pn

        self.stick = stick
        self.multiplier = multiplier

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.results = Results(stress=False, elasticity=False)

        self.results.dx_ref = np.zeros_like(self.field.fields[0].values[self.points])
        self.results.active = np.zeros(len(self.points), dtype=bool)

    def plot(
        self,
        plotter=None,
        offset=0.0,
        show_edges=True,
        color="black",
        opacity=0.5,
        deformed=True,
        size=None,
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
        sym_mask = np.array(sym, dtype=bool)
        plane.scale(~sym_mask / 2 + 0.5, inplace=True)
        plane.translate(sym_mask * size / 4, inplace=True)

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
                **kwargs,
            )

        return plotter

    def _vector(self, field=None, parallel=False):

        if field is not None:
            self.field = field

        # displacement field values at mesh points
        u = self.field.fields[0].values

        # deformed coordinates of center-point and points in potential contact
        xc = u[self.centerpoint] + self.mesh.points[self.centerpoint]
        x = u[self.points] + self.mesh.points[self.points]

        # gap vectors
        dx = x - xc
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

            if self.stick:  # tangential stick contribution
                dx_delta = dx[contact] - self.results.dx_ref[contact]
                force_t = self.multiplier * (dx_delta @ self.Pt)

                r[self.points[contact]] += force_t

        self.results.force = r.reshape(-1, 1).tocsr()

        return self.results.force

    def _matrix(self, field=None, parallel=False):

        if field is not None:
            self.field = field

        # displacement field values at mesh points
        u = self.field.fields[0].values

        # deformed coordinates of center-point and points in potential contact
        xc = u[self.centerpoint] + self.mesh.points[self.centerpoint]
        x = u[self.points] + self.mesh.points[self.points]

        # gap vectors
        dx = x - xc
        gap = dx @ self.normal
        contact_mask = gap < 0.0
        contact = np.where(contact_mask)[0]

        L = lil_matrix((self.mesh.ndof, self.mesh.ndof))

        if len(contact) > 0:

            indices = self.field[0].indices.dof
            t = indices[self.points[contact]]
            c = indices[self.centerpoint]

            # normal contribution
            K_n = self.multiplier * self.Pn

            if self.stick:  # tangential contribution
                K_t = self.multiplier * self.Pt

            # global assembly
            for p in range(len(contact)):

                # normal contribution
                L[t[p].reshape(-1, 1), t[p]] += K_n
                L[t[p].reshape(-1, 1), c] -= K_n
                L[c.reshape(-1, 1), t[p]] -= K_n
                L[c.reshape(-1, 1), c] += K_n

                if self.stick:  # tangential contribution
                    L[t[p].reshape(-1, 1), t[p]] += K_t

        self.results.stiffness = L.tocsr()

        return self.results.stiffness

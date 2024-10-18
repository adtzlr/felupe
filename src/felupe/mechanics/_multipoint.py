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
from scipy.sparse import eye, lil_matrix

from ._helpers import Assemble, Results


class MultiPointConstraint:
    """A Multi-point-constraint which connects a center-point to a list of points.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray
        An array with indices of points to be connected to the center-point.
    centerpoint : int
        The index of the centerpoint.
    skip : 3-tuple of bool, optional
        A tuple with boolean values for each axis to skip. If True, the respective axis
        is not connected. Default is (False, False, False).
    multiplier : float, optional
        A multiplier to penalize the relative displacements between the center-point and
        the points. Default is 1e-3.

    Notes
    -----
    A :class:`~felupe.MultiPointConstraint` is supported as an item in a
    :class:`~felupe.Step`. It provides the assemble-methods
    :meth:`MultiPointConstraint.assemble.vector() <felupe.MultiPointConstraint.assemble.vector>`
    and :meth:`MultiPointConstraint.assemble.matrix() <felupe.MultiPointConstraint.assemble.matrix>`.

    ..  note::

        Rotational degrees-of-freedom of the center-point are not connected to the
        points.

    Examples
    --------
    This example shows how to use a :class:`~felupe.MultiPointConstraint`.

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

    A :class:`~felupe.MultiPointConstraint` defines the multi-point constraint which
    connects the displacement degrees of freedom of the center-point with the dofs of
    points located at :math:`x=1`.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> mpc = fem.MultiPointConstraint(
        ...     field=field,
        ...     points=np.arange(mesh.npoints)[mesh.x == 1],
        ...     centerpoint=-1,
        ... )
        >>>
        >>> plotter = pv.Plotter()
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[mpc.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[mpc.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> mesh.plot(plotter=mpc.plot(plotter=plotter)).show()

    The mesh is fixed on the left end face and a ramped :class:`~felupe.PointLoad` is
    applied on the center-point of the :class:`~felupe.MultiPointConstraint`. All items
    are added to a :class:`~felupe.Step` and a :class:`~felupe.Job` is evaluated.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {"fixed": fem.Boundary(displacement, fx=0)}
        >>> load = fem.PointLoad(field, points=[-1])
        >>> table = fem.math.linsteps([0, 1], num=5, axis=0, axes=3)
        >>>
        >>> step = fem.Step(
        ...     [solid, mpc, load], boundaries=boundaries, ramp={load: table}
        ... )
        >>> job = fem.Job([step]).evaluate()

    A view on the deformed mesh including the :class:`~felupe.MultiPointConstraint` is
    plotted.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> plotter = pv.Plotter()
        >>>
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[mpc.points] + displacement.values[mpc.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[mpc.centerpoint]] + displacement.values[[mpc.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> field.plot(
        ...     "Displacement", component=None, plotter=mpc.plot(plotter=plotter)
        ... ).show()

    See Also
    --------
    felupe.MultiPointContact : A frictionless point-to-rigid (wall) contact.

    """

    def __init__(
        self, field, points, centerpoint, skip=(False, False, False), multiplier=1e3
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.asarray(points)
        self.centerpoint = centerpoint
        self.mask = ~np.array(skip, dtype=bool)[: self.mesh.dim]
        self.axes = np.arange(self.mesh.dim)[self.mask]
        self.multiplier = multiplier

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def plot(self, plotter=None, color="black", **kwargs):
        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()

        # get deformed points
        x = self.mesh.points + self.field[0].values
        x = np.pad(x, ((0, 0), (0, 3 - x.shape[1])))
        pointa = x[self.centerpoint]

        for pointb in x[self.points]:
            plotter.add_mesh(pv.Line(pointa, pointb), color=color, **kwargs)

        return plotter

    def _vector(self, field=None, parallel=False):
        "Calculate vector of residuals with RBE2 contributions."

        if field is not None:
            self.field = field

        u = self.field.fields[0].values
        N = self.multiplier * (-u[self.points] + u[self.centerpoint])
        N[:, ~self.mask] = 0

        r = lil_matrix(u.shape)
        r[self.points] = -N
        r[self.centerpoint] = N.sum(axis=0)

        self.results.force = r.reshape(-1, 1).tocsr()
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate stiffness with RBE2 contributions."

        if field is not None:
            self.field = field

        indices = np.arange(self.mesh.ndof).reshape(self.mesh.points.shape)
        td = [indices[self.points.reshape(-1, 1), ax].ravel() for ax in self.axes]
        cd = [indices[self.centerpoint, ax].ravel() for ax in self.axes]

        L = lil_matrix((self.mesh.ndof, self.mesh.ndof))

        for t, c in zip(td, cd):
            L[t.reshape(-1, 1), t] = eye(len(t)) * self.multiplier
            L[t.reshape(-1, 1), c] = -self.multiplier
            L[c.reshape(-1, 1), t] = -self.multiplier
            L[c.reshape(-1, 1), c] = eye(len(c)) * self.multiplier * len(self.points)

        self.results.stiffness = L.tocsr()
        return self.results.stiffness


class MultiPointContact:
    """A frictionless point-to-rigid (wall) contact which connects a center-point to a
    list of points.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray
        An array with indices of points to be connected to the center-point.
    centerpoint : int
        The index of the centerpoint.
    skip : 3-tuple of bool, optional
        A tuple with boolean values for each axis to skip. If True, the respective axis
        is not connected. Default is (False, False, False).
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
        ...     mesh.points[mpc.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[mpc.centerpoint]],
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
        >>> field.plot(
        ...     "Displacement", component=None, plotter=contact.plot(plotter=plotter)
        ... ).show()

    See Also
    --------
    felupe.MultiPointConstraint : A Multi-point-constraint which connects a center-point
        to a list of points.

    """

    def __init__(
        self, field, points, centerpoint, skip=(False, False, False), multiplier=1e6
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.asarray(points)
        self.centerpoint = centerpoint
        self.mask = ~np.array(skip, dtype=bool)[: self.mesh.dim]
        self.axes = np.arange(self.mesh.dim)[self.mask]
        self.multiplier = multiplier

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def plot(
        self,
        plotter=None,
        offset=0,
        show_edges=True,
        color="black",
        opacity=0.5,
        **kwargs,
    ):
        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()

        # get edge lengths of deformed enclosing box
        x = self.mesh.points + self.field[0].values
        edges = np.diag((x.max(axis=0) - x.min(axis=0))) + x.min(axis=0)

        # plot a line or a rectangle for each active contact plane
        for ax in self.axes:
            # fill the point values of the normal axis with the centerpoint values
            points = edges.copy()
            points[:, ax] = x[self.centerpoint, ax] + offset

            # scale the line or rectangle at the origin
            origin = points.mean(axis=0)
            points = (points - origin) * 1.05 + origin

            # plot a line or a rectangle
            if len(points) == 3:
                plotter.add_mesh(
                    pv.Rectangle(points), color=color, opacity=opacity, **kwargs
                )
            else:
                points = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
                plotter.add_mesh(
                    pv.Line(*points), color=color, opacity=opacity, **kwargs
                )

        return plotter

    def _vector(self, field=None, parallel=False):
        "Calculate vector of residuals with RBE2 contributions."

        if field is not None:
            self.field = field

        u = self.field.fields[0].values

        Xc = self.mesh.points[self.centerpoint]
        Xt = self.mesh.points[self.points]

        xc = u[self.centerpoint] + Xc
        xt = u[self.points] + Xt

        mask = np.sign(-Xt + Xc) == np.sign(-xt + xc)
        mask[:, ~self.mask] = True
        n = -xt + xc
        n[mask] = 0

        r = lil_matrix(u.shape)
        r[self.points] = -self.multiplier * n
        r[self.centerpoint] = self.multiplier * n.sum(axis=0)

        self.results.force = r.reshape(-1, 1).tocsr()
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate stiffness with RBE2 contributions."

        if field is not None:
            self.field = field

        u = self.field.fields[0].values

        Xc = self.mesh.points[self.centerpoint]
        Xt = self.mesh.points[self.points]

        xc = u[self.centerpoint] + Xc
        xt = u[self.points] + Xt

        mask = np.sign(-Xt + Xc) != np.sign(-xt + xc)
        masks = [mask[:, ax] for ax in self.axes]

        indices = np.arange(self.mesh.ndof).reshape(self.mesh.points.shape)
        td = [indices[self.points.reshape(-1, 1), ax].ravel() for ax in self.axes]
        cd = [indices[self.centerpoint, ax].ravel() for ax in self.axes]

        L = lil_matrix((self.mesh.ndof, self.mesh.ndof))

        for t, c, m in zip(td, cd, masks):
            L[t[m].reshape(-1, 1), t[m]] = eye(len(t[m])) * self.multiplier
            L[t[m].reshape(-1, 1), c] = -self.multiplier
            L[c.reshape(-1, 1), t[m]] = -self.multiplier
            L[c.reshape(-1, 1), c] = eye(len(c)) * self.multiplier * len(self.points[m])

        self.results.stiffness = L.tocsr()
        return self.results.stiffness

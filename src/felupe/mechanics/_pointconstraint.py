# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..view._backends import backend_of
from ._helpers import Assemble, Results


class PointConstraint:
    r"""Base class of penalty-based constraints of the displacements of points in a
    coordinate system, e.g. a :class:`~felupe.CartesianConstraint` or a
    :class:`~felupe.CylindricalConstraint`.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray of int or bool
        An array with indices (or a mask) of the points to be constrained.
    displacement : array_like
        The prescribed displacements in the three directions of the coordinate system,
        for all points or per point with shape (n, 3).
    skip : tuple of bool
        Flags to skip the directions of the coordinate system.
    multiplier : float
        The penalty stiffness :math:`k` per point and direction.

    Notes
    -----
    A subclass must implement the method ``_gaps(u, stiffness=True)``, which returns
    the gaps :math:`g_i` with shape (n, 3), their gradients
    :math:`\partial g_i / \partial \boldsymbol{x}` w.r.t. the deformed coordinates with
    shape (n, 3, 3) as (point, direction, component) and the sum of the gap-weighted
    Hessians :math:`\sum_i g_i\, \partial^2 g_i / (\partial \boldsymbol{x}\, \partial
    \boldsymbol{x})` of the constrained directions with shape (n, 3, 3) or None (if all
    gaps are linear or ``stiffness=False``) for the padded displacements
    :math:`\boldsymbol{u}` of the points with shape (n, 3). The vector and the matrix
    of the penalty potential :math:`\Pi = \sum_{\text{points}} \sum_i \frac{k}{2}\,
    g_i^2` are assembled by this class, where the skipped directions do not
    contribute.
    """

    def __init__(self, field, points, displacement, skip, multiplier):
        self.field = field
        self.mesh = field.region.mesh
        self.dim = self.mesh.dim
        self.points = np.array(points)
        self.multiplier = multiplier

        if self.dim not in [2, 3]:
            raise ValueError(f"A {type(self).__name__} requires a 2d- or a 3d-mesh.")

        if len(self.points) == 0:
            self.points = self.points.astype(int)

        if self.points.dtype == bool:
            self.points = np.where(self.points)[0]

        # mask of the constrained directions
        skip = np.asarray(skip, dtype=bool).ravel()
        skip = np.pad(skip, (0, 3 - len(skip)), constant_values=True)
        self.mask = ~skip

        if self.dim == 2:
            self.mask[2] = False

        self.displacement = None
        self.update(displacement)

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def _pad(self, values):
        "Pad the last axis of an array with zeros to three components."
        width = [(0, 0)] * (values.ndim - 1) + [(0, 3 - values.shape[-1])]
        return np.pad(values, width)

    def update(self, displacement):
        r"""Update the prescribed displacements.

        Parameters
        ----------
        displacement : array_like
            The prescribed displacements in the three directions of the coordinate
            system, either one vector for all points or one vector per point with
            shape (n, 3). For 2d-meshes, also vectors with two components are
            supported.

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
        gap, dgdx, hessian = self._gaps(u, stiffness=stiffness)

        # the skipped directions do not contribute
        gap = gap * self.mask
        dgdx = dgdx * self.mask.reshape(-1, 1)

        self.results.gap = gap

        force = np.einsum("ai,aij->aj", gap, dgdx)
        force = self.multiplier * force[:, : self.dim]
        matrix = None

        if stiffness:
            matrix = np.einsum("aij,aik->ajk", dgdx, dgdx)

            if hessian is not None:
                matrix += hessian

            matrix = self.multiplier * matrix[:, : self.dim, : self.dim]

        return force, matrix

    def _vector(self, field=None, parallel=False):
        "Calculate the vector of residuals of the constraint."

        force, _ = self._evaluate(field, stiffness=False)

        dof = self.field[0].indices.dof[self.points].ravel()
        ndof = self.field[0].values.size

        self.results.force = csr_matrix(
            (force.ravel(), (dof, np.zeros_like(dof))), shape=(ndof, 1)
        )
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate the stiffness matrix of the constraint."

        _, matrix = self._evaluate(field, stiffness=True)

        dof = self.field[0].indices.dof[self.points]
        ndof = self.field[0].values.size

        rows = np.repeat(dof, self.dim, axis=1).ravel()
        cols = np.tile(dof, (1, self.dim)).ravel()

        self.results.stiffness = coo_matrix(
            (matrix.ravel(), (rows, cols)), shape=(ndof, ndof)
        ).tocsr()
        return self.results.stiffness

    def _plot_reference(self, plotter, x, color, line_width):
        """Add the reference geometry of the coordinate system to the plotter. By
        default, nothing is added."""

    def plot(
        self,
        plotter=None,
        color="black",
        deformed=True,
        point_size=10,
        line_width=2,
        **kwargs,
    ):
        """Plot the constrained points and the coordinate system of the constraint.

        Parameters
        ----------
        plotter : pyvista.Plotter or None, optional
            An existing plotter. If None, a new plotter is created. Default is None.
        color : str, optional
            The color of the points and the coordinate system. Default is "black".
        deformed : bool, optional
            A flag to plot the points in the deformed configuration. Default is True.
        point_size : float, optional
            The size of the points. Default is 10.
        line_width : float, optional
            The line width of the coordinate system. Default is 2.
        **kwargs : dict, optional
            Additional keyword arguments for the points, passed to
            :meth:`pyvista.Plotter.add_points` (by the plotting backend of PyVista).

        Returns
        -------
        pyvista.Plotter
            The plotter with the points and the coordinate system.
        """

        backend = backend_of(plotter)

        if plotter is None:
            plotter = backend.prepare_plotter(theme=None)

        x = self.mesh.points[self.points]

        if deformed:
            x = x + self.field[0].values[self.points]

        x = self._pad(x)

        if len(x) > 0:
            plotter = backend.add_points(
                plotter, x, color=color, point_size=point_size, **kwargs
            )
            self._plot_reference(plotter, x, color=color, line_width=line_width)

        return plotter

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
from scipy.sparse import csr_matrix

from ._helpers import Assemble, Results


class PointLoad:
    r"""A point load with methods for the assembly of sparse vectors/matrices, applied
    on the n-th field.

    Parameters
    ----------
    field : FieldContainer
        A field container with fields created on a region.
    points : list of int
        A list with point ids where the values are applied.
    values : float or array_like or None, optional
        Values at points (default is None). If None, the values are set to zero.
    apply_on : int, optional
        The n-th field on which the point load is applied (default is 0).
    axisymmetric : bool, optional
        A flag to multiply the assembled vector and matrix by a scaling factor of
        :math:`2 \pi` (default is False).

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.mesh.Line(n=3)
        >>> element = fem.element.Line()
        >>> quadrature = fem.GaussLegendre(order=1, dim=1)
        >>>
        >>> region = fem.Region(mesh, element, quadrature)
        >>> field = fem.FieldContainer([fem.Field(region, dim=1)])
        >>>
        >>> load = fem.PointLoad(field, [1, 2], values=[[3], [5]])
        >>>
        >>> vector = load.assemble.vector()
        >>> vector.toarray()
        array([[0.],
               [3.],
               [5.]])
    """

    def __init__(self, field, points, values=None, apply_on=0, axisymmetric=False):
        self.field = field
        self.points = points

        if values is None:
            self.values = 0
        else:
            self.values = values

        self.apply_on = apply_on
        self.axisymmetric = axisymmetric

        self.results = Results()
        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )

    def update(self, values):
        self.__init__(self.field, self.points, values, self.apply_on, self.axisymmetric)

    def _vector(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        force = [np.zeros_like(f.values) for f in self.field.fields]
        force[self.apply_on][self.points] += self.values

        if self.axisymmetric:
            mesh_points = self.field[0].region.mesh.points
            radius = mesh_points[self.points, 1].reshape(-1, 1)
            force[self.apply_on][self.points] *= 2 * np.pi * radius

        self.results.force = csr_matrix(
            np.concatenate([f.ravel() for f in force]).reshape(-1, 1)
        )

        return self.results.force

    def _matrix(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        n = np.sum(self.field.fieldsizes)
        self.results.stiffness = csr_matrix(([0.0], ([0], [0])), shape=(n, n))

        return self.results.stiffness

    def plot(
        self,
        plotter=None,
        color="red",
        scale=0.125,
        **kwargs,
    ):
        "Plot the point load."

        mesh = self.field.region.mesh

        if plotter is None:
            plotter = mesh.plot()

        if len(self.points) > 0:
            points = np.pad(mesh.points, ((0, 0), (0, 3 - mesh.dim)))
            magnitude = min(mesh.points.max(axis=0) - mesh.points.min(axis=0)) * scale

            values = np.atleast_2d(self.values)

            skip = np.zeros(3, dtype=bool)
            skip[values.shape[1] :] = True

            if values.shape[1] > 1:
                skip[:values.shape[1]][np.isclose(values, 0).all(axis=0)] = True

            for a, (skip_axis, direction) in enumerate(zip(skip, np.eye(3))):
                d = np.broadcast_to(direction.reshape(1, 3), points[self.points].shape)
                if not skip_axis:
                    _ = plotter.add_arrows(
                        points[self.points],
                        direction=d * np.sign(values[0, a]),
                        mag=magnitude,
                        color=color,
                        **kwargs,
                    )

        return plotter

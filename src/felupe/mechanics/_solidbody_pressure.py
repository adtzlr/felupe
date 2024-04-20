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

from ..assembly import IntegralForm
from ..constitution import AreaChange
from ._helpers import Assemble, Results


class SolidBodyPressure:
    r"""A hydrostatic pressure boundary on a solid body.

    Parameters
    ----------
    field : FieldContainer
        A field container with fields created on a boundary region.
    pressure : float or ndarray or None, optional
        A scaling factor for the prescribed pressure :math:`p` (default is None). If
        None, the pressure is set to zero.

    Notes
    -----
    ..  math::

        \delta W_{ext} = \int_{\partial V}
            \delta \boldsymbol{u} \cdot p J \boldsymbol{F}^{-T} \ d\boldsymbol{A}

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=6)
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
        >>> boundaries = fem.dof.symmetry(field[0])
        >>> umat = fem.NeoHooke(mu=1, bulk=2)
        >>> solid = fem.SolidBody(umat, field)
        >>>
        >>> region_pressure = fem.RegionQuadBoundary(
        ...     mesh=mesh,
        ...     only_surface=True,  # select only faces on the outline
        ...     mask=mesh.points[:, 0] == 1,  # select a subset of faces on the surface
        ...     ensure_3d=True,  # requires True for axisymmetric/plane strain, otherwise False
        ... )
        >>> field_boundary = fem.FieldContainer([fem.FieldAxisymmetric(region_pressure, dim=2)])
        >>> pressure = fem.SolidBodyPressure(field=field_boundary)
        >>>
        >>> table = fem.math.linsteps([0, 1], num=5)
        >>> step = fem.Step(
        ...     items=[solid, pressure], ramp={pressure: 1 * table}, boundaries=boundaries
        ... )
        >>>
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> solid.plot(
        ...     "Principal Values of Cauchy Stress", component=2, clim=[-1.01, -0.99]
        ... ).show()
    """

    def __init__(self, field, pressure=None):
        self.field = field
        self._normals = self.field.region.normals

        self.results = Results()
        self.results.kinematics = self._extract(self.field)

        self.results.pressure = 0
        if pressure is not None:
            self.results.pressure = pressure

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self._area_change = AreaChange()

    def update(self, pressure):
        self.__init__(self.field, pressure)

    def _extract(self, field):
        self.field = field
        self.results.kinematics = self.field.extract()

        return self.results.kinematics

    def _vector(self, field=None, pressure=None, parallel=False, resize=None):
        if field is not None:
            self._update(field)
            self.results.kinematics = self._extract(self.field)

        fun = self._area_change.function(
            self.results.kinematics,
            self._normals,
            parallel=parallel,
        )

        if pressure is not None:
            self.results.pressure = pressure

        fun[0] *= self.results.pressure

        self.results.force = IntegralForm(
            fun=fun, v=self.field, dV=self.field.region.dV, grad_v=[False]
        ).assemble(parallel=parallel)

        if resize is not None:
            self.results.force.resize(*resize.shape)

        return self.results.force

    def _matrix(self, field=None, pressure=None, parallel=False, resize=None):
        if field is not None:
            self._update(field)
            self.results.kinematics = self._extract(self.field)

        fun = self._area_change.gradient(
            self.results.kinematics,
            self._normals,
            parallel=parallel,
        )

        if pressure is not None:
            self.results.pressure = pressure

        fun[0] *= self.results.pressure

        self.results.stiffness = IntegralForm(
            fun=fun,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
            grad_v=[False],
            grad_u=[True],
        ).assemble(parallel=parallel)

        if resize is not None:
            self.results.stiffness.resize(*resize.shape)

        return self.results.stiffness

    def _update(self, other_field, field=None):
        if field is not None:
            self.field = field

        self.field[0].values = other_field[0].values
        self.results.kinematics = self._extract(self.field)

        return self.field

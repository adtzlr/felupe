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

from ..assembly import IntegralForm
from ..constitution import AreaChange
from ..math import dot
from ._helpers import Assemble, Results


class SolidBodyCauchyStress:
    r"""A Cauchy stress boundary on a solid body.

    Parameters
    ----------
    field : FieldContainer
        A field container with fields created on a boundary region.
    stress : ndarray of shape (3, 3, ...) or None, optional
        The prescribed Cauchy stress components :math:`\sigma_{ij}` (default is None).
        If None, all Cauchy stress components are set to zero.

    Notes
    -----

    ..  math::

        \delta W_{ext} = \int_{\partial V}
            \delta \boldsymbol{u} \cdot \boldsymbol{\sigma}\ d\boldsymbol{a} =
            \int_{\partial V}
            \delta \boldsymbol{u} \cdot \boldsymbol{\sigma} J \boldsymbol{F}^{-T}\
            d\boldsymbol{A}

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=6)
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
        >>>
        >>> boundaries = {"fixed": fem.Boundary(field[0], fx=0)}
        >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=2), field=field)
        >>>
        >>> mask = np.logical_and(mesh.x == 1, mesh.y > 0.5)
        >>> region_stress = fem.RegionQuadBoundary(
        ...     mesh=mesh,
        ...     mask=mask,  # select a subset of faces on the surface
        ...     ensure_3d=True,  # True for axisymmetric/plane strain
        ... )
        >>> field_boundary = fem.FieldContainer([fem.FieldAxisymmetric(region_stress, dim=2)])
        >>> stress = fem.SolidBodyCauchyStress(field=field_boundary)
        >>>
        >>> table = (
        ...     fem.math.linsteps([0, 1], num=5, axis=1, axes=9)
        ...     + fem.math.linsteps([0, 1], num=5, axis=3, axes=9)
        ... ).reshape(-1, 3, 3)
        >>>
        >>> step = fem.Step(
        ...     items=[solid, stress], ramp={stress: 1.0 * table}, boundaries=boundaries
        ... )
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> solid.plot("Principal Values of Cauchy Stress").show()
    """

    def __init__(self, field, cauchy_stress=None):
        self.field = field
        self._normals = self.field.region.normals

        self.results = Results()
        self.results.kinematics = self._extract(self.field)

        self.results.cauchy_stress = np.zeros((3, 3))
        if cauchy_stress is not None:
            self.results.cauchy_stress = cauchy_stress

        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )
        self._area_change = AreaChange()

    def update(self, cauchy_stress):
        self.__init__(self.field, cauchy_stress)

    def _extract(self, field):
        self.field = field
        self.results.kinematics = self.field.extract()

        return self.results.kinematics

    def _vector(self, field=None, parallel=False, resize=None):
        if field is not None:
            self._update(field)
            self.results.kinematics = self._extract(self.field)

        fun = self._area_change.function(
            self.results.kinematics,
            self._normals,
            parallel=parallel,
        )

        fun[0] = dot(self.results.cauchy_stress, fun[0], mode=(2, 2), out=fun[0])

        self.results.force = IntegralForm(
            fun=fun, v=self.field, dV=self.field.region.dV, grad_v=[False]
        ).assemble(parallel=parallel)

        if resize is not None:
            self.results.force.resize(*resize.shape)

        return self.results.force

    def _matrix(self, field=None, parallel=False, resize=None):
        if field is not None:
            self._update(field)
            self.results.kinematics = self._extract(self.field)

        fun = self._area_change.gradient(
            self.results.kinematics,
            self._normals,
            parallel=parallel,
        )

        fun[0] = dot(self.results.cauchy_stress, fun[0], mode=(2, 4), out=fun[0])

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

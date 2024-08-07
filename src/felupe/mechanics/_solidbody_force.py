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

from ..assembly import IntegralForm
from ._helpers import Assemble, Results


class SolidBodyForce:
    r"""A body force on a solid body.

    Parameters
    ----------
    field : FieldContainer
        A field container with fields created on a boundary region.
    values : ndarray or None, optional
        The prescribed values (e.g. gravity :math:`\boldsymbol{g}`). Default is None. If
        None, the values are set to zero (the dimension is derived from the first field
        of the field container).
    scale : float, optional
        An optional scale factor for the values, e.g. density :math:`\rho` of the solid
        body. Default is 1.0.

    Notes
    -----
    ..  math::

        \delta W_{ext} = \int_V
            \delta \boldsymbol{u} \cdot \rho \boldsymbol{g} \ dV

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> boundaries = fem.dof.symmetry(field[0])
        >>>
        >>> umat = fem.NeoHooke(mu=1, bulk=2)
        >>> solid = fem.SolidBody(umat, field)
        >>> density = 1.0
        >>> force = fem.SolidBodyForce(field, scale=density)
        >>>
        >>> gravity = fem.math.linsteps([0, 2], num=5, axis=0, axes=3)
        >>> step = fem.Step(
        ...     items=[solid, force],
        ...     ramp={force: gravity},
        ...     boundaries=boundaries,
        ... )
        >>>
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> solid.plot("Principal Values of Cauchy Stress").show()

    """

    def __init__(self, field, values=None, scale=1.0):
        self.field = field
        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, multiplier=-1.0
        )
        self._form = IntegralForm

        self.results.values = np.zeros(self.field[0].dim)
        if values is not None:
            self.results.values = np.array(values)

        self.results.scale = scale

    def update(self, values):
        self.__init__(self.field, values, self.results.scale)

    def _vector(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        # copy and take only the first (displacement) field of the container
        f = self.field.copy()
        f.fields = f.fields[0:1]

        self.results.force = self._form(
            fun=[self.results.scale * self.results.values.reshape(-1, 1, 1)],
            v=f,
            dV=self.field.region.dV,
            grad_v=[False],
        ).assemble(parallel=parallel)

        if len(self.field) > 1:
            self.results.force.resize(np.sum(self.field.fieldsizes), 1)

        return self.results.force

    def _matrix(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        n = np.sum(self.field.fieldsizes)
        self.results.stiffness = csr_matrix(([0.0], ([0], [0])), shape=(n, n))

        return self.results.stiffness

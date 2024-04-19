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


class SolidBodyGravity:
    r"""A gravity (body) force on a solid body.

    Parameters
    ----------
    field : FieldContainer
        A field container with fields created on a boundary region.
    gravity : ndarray or None, optional
        The prescribed values of gravity :math:`\boldsymbol{g}` (default is None). If
        None, the gravity vector is set to zero (the dimension of the gravity vector is
        derived from the first field of the field container).
    density : float, optional
        The density :math:`\rho` of the solid body (default is 1.0).

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
        >>> gravity = fem.SolidBodyGravity(field, density=1.0)
        >>>
        >>> table = fem.math.linsteps([0, 1], num=5, axis=0, axes=3)
        >>> step = fem.Step(
        ...     items=[solid, gravity],
        ...     ramp={gravity: 2 * table},
        ...     boundaries=boundaries,
        ... )
        >>>
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> solid.plot("Principal Values of Cauchy Stress").show()

    """

    def __init__(self, field, gravity=None, density=1.0):
        self.field = field
        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self._form = IntegralForm

        self.results.gravity = np.zeros(self.field[0].dim)
        if gravity is not None:
            self.results.gravity = np.array(gravity)

        self.results.density = density

    def update(self, gravity):
        self.__init__(self.field, gravity, self.results.density)

    def _vector(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        # copy and take only the first (displacement) field of the container
        f = self.field.copy()
        f.fields = f.fields[0:1]

        self.results.force = self._form(
            fun=[self.results.density * self.results.gravity.reshape(-1, 1, 1)],
            v=f,
            dV=self.field.region.dV,
            grad_v=[False],
        ).assemble(parallel=parallel)

        if len(self.field) > 1:
            self.results.force.resize(np.sum(self.field.fieldsizes), 1)

        return -self.results.force

    def _matrix(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        n = np.sum(self.field.fieldsizes)
        self.results.stiffness = csr_matrix(([0], ([0], [0])), shape=(n, n))

        return self.results.stiffness

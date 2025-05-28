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


import inspect

import numpy as np

from ..assembly import IntegralForm
from ..math import dot, dya, identity, sqrt
from ._helpers import Assemble, Evaluate, Results
from ._solidbody import Solid


class TrussBody(Solid):
    r"""A Truss body with methods for the assembly of sparse vectors/matrices.

    Parameters
    ----------
    umat : class
        A class which provides methods for evaluating the gradient and the hessian of
        the strain energy density function per unit undeformed volume. The function
        signatures must be ``dψdλ, ζ_new = umat.gradient([λ, ζ])`` for
        the gradient and ``d2ψdλdλ = umat.hessian([λ, ζ])`` for the hessian of
        the strain energy density function :math:`\psi(\lambda)`, where
        :math:`\lambda` is the stretch and :math:`\zeta`
        holds the array of internal state variables.
    field : FieldContainer
        A field container with one or more fields.
    area : float or ndarray
        The cross-sectional areas of the trusses.
    statevars : ndarray or None, optional
        Array of initial internal state variables (default is None).

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Mesh(
        ...     points=[[0, 0], [1, 1], [2.0, 0]], cells=[[0, 1], [1, 2]], cell_type="line"
        ... )
        >>> region = fem.Region(mesh, fem.Line(), fem.GaussLobatto(order=0, dim=1), grad=False)
        >>> field = fem.Field(region, dim=2).as_container()
        >>> boundaries = fem.BoundaryDict(fixed=fem.Boundary(field[0], fy=0))
        >>>
        >>> umat = fem.LinearElastic1D(E=[1, 1])
        >>> truss = fem.TrussBody(umat, field, area=[1, 1])
        >>> load = fem.PointLoad(field, [1])
        >>>
        >>> mesh.plot(plotter=load.plot(plotter=boundaries.plot()), line_width=5).show()
        >>>
        >>> move = fem.math.linsteps([0, -0.1], num=5, axis=1, axes=2)
        >>> step = fem.Step(items=[truss, load], ramp={load: move}, boundaries=boundaries)
        >>> job = fem.Job(steps=[step]).evaluate()

    See Also
    --------
    felupe.SolidBodyNearlyIncompressible : A (nearly) incompressible solid body with
        methods for the assembly of sparse vectors/matrices.
    """

    def __init__(self, umat, field, area, statevars=None):
        self.field = field
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.evaluate = Evaluate(gradient=self._gradient, hessian=self._hessian)
        self.results = Results(stress=True, elasticity=True)

        self.umat = umat
        self.area = np.array(area)

        self.form_vector = IntegralForm(
            [None],
            v=self.field,
            dV=None,
            grad_v=[False],
        )
        self.form_matrix = IntegralForm(
            [None],
            v=self.field,
            dV=None,
            u=self.field,
            grad_v=[False],
            grad_u=[False],
        )

        if statevars is not None:
            self.results.statevars = statevars
        else:
            statevars_shape = (0,)
            if hasattr(umat, "x"):
                statevars_shape = umat.x[-1].shape
            self.results.statevars = np.zeros(
                (
                    *statevars_shape,
                    field.region.mesh.ncells,
                )
            )

        cells = self.field.region.mesh.cells
        self.X = self.field.region.mesh.points[cells].T
        dX = self.X[:, 1] - self.X[:, 0]
        self.length_undeformed = sqrt(dot(dX, dX, mode=(1, 1)))

    def _kinematics(self, field):

        cells = self.field.region.mesh.cells
        u = field[0].values[cells].T
        x = self.X + u

        dx = x[:, 1] - x[:, 0]

        length_deformed = sqrt(dot(dx, dx, mode=(1, 1)))

        stretch = length_deformed / self.length_undeformed
        normal_deformed = dx / length_deformed

        return stretch, length_deformed, normal_deformed

    def _gradient(self, field=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.field = field
            self.results.kinematics = self._kinematics(field)

        if "out" in inspect.signature(self.umat.gradient).parameters:
            kwargs["out"] = self.results.gradient

        gradient = self.umat.gradient(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )
        self.results.gradient = self.results.stress = gradient[0]
        self.results._statevars = gradient[-1]

        return self.results.gradient

    def _hessian(self, field=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.field = field
            self.results.kinematics = self._kinematics(field)

        if "out" in inspect.signature(self.umat.hessian).parameters:
            kwargs["out"] = self.results.hessian

        hessian = self.umat.hessian(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )
        self.results.hessian = self.results.elasticity = hessian[0]

        return self.results.hessian

    def _vector(self, field=None, parallel=None, args=(), kwargs=None):

        self.results.stress = self._gradient(field, args=args, kwargs=kwargs)
        normal_deformed = self.results.kinematics[2]

        force = self.results.stress * self.area
        r = force * np.array([-normal_deformed, normal_deformed])  # (a, i, cell)

        self.results.force = self.form_vector.assemble(values=[r], parallel=parallel)

        return self.results.force

    def _matrix(self, field=None, parallel=None, args=(), kwargs=None):

        self.results.hessian = self._hessian(field, args=args, kwargs=kwargs)
        L = self.length_undeformed
        stretch, l, n = self.results.kinematics

        S = self.results.stress = self._gradient(field, args=args, kwargs=kwargs)
        dSdE = self.results.elasticity = self._hessian(field, args=args, kwargs=kwargs)

        m = dya(n, n, mode=1)
        eye = identity(dim=len(n), shape=(1,))
        K_EE = dSdE / L * self.area * m + S / l * self.area * (eye - m)

        K = np.stack([[K_EE, -K_EE], [-K_EE, K_EE]])  # (a, b, i, j, cell)

        self.results.stiffness = self.form_matrix.assemble(
            values=[K.transpose([0, 2, 1, 3, 4])], parallel=parallel
        )

        return self.results.stiffness

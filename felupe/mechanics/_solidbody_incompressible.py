# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from ._helpers import StateNearlyIncompressible
from .._assembly import IntegralFormMixed
from .._field import FieldAxisymmetric
from ..constitution import AreaChange
from ..math import dot, transpose, det, dya, ddot
from ._helpers import Assemble, Evaluate, Results


class SolidBodyNearlyIncompressible:
    r"""A (nearly) incompressible SolidBody with methods for the assembly of
    sparse vectors/matrices based on a ``MaterialTensor`` with state variables.
    
    The volumetric material behaviour is defined by a strain energy function.
    
    ..  math::
        
        U(J) = \frac{K}{2} (J - 1)^2
    
    """

    def __init__(self, umat, field, bulk, state=None, statevars=None):
        """A (nearly) incompressible SolidBody with methods for the assembly of
        sparse vectors/matrices.

        Parameters
        ----------
        umat : A constitutive material formulation with methods for the evaluation
            of the gradient ``P = umat.gradient(F)`` as well as the hessian
            ``A = umat.hessian(F)`` of the strain energy function w.r.t. the
            deformation gradient.
        field : FieldContainer
            The field (and its underlying region) on which the solid body will
            be created on.
        bulk : float
            The bulk modulus of the volumetric material behaviour
            (:math:`U(J)=K(J-1)^2/2`).
        state : StateNearlyIncompressible
            A valid initial state for a (nearly) incompressible solid.
        """

        self.umat = umat
        self.field = field
        self.bulk = bulk

        self._area_change = AreaChange()
        self._form = IntegralFormMixed

        # volume of undeformed configuration
        if isinstance(self.field[0], FieldAxisymmetric):
            R = self.field[0].radius
            dA = self.field.region.dV
            dV = 2 * np.pi * R * dA
        else:
            dV = self.field.region.dV
        self.V = dV.sum(0)

        self.results = Results(stress=True, elasticity=True)

        if statevars is not None:
            self.results.statevars = statevars
        else:
            self.results.statevars = np.zeros(
                (
                    *umat.x[-1].shape,
                    field.region.quadrature.npoints,
                    field.region.mesh.ncells,
                )
            )

        if state is None:
            # init state of internal fields
            self.results.state = StateNearlyIncompressible(field)
        else:
            self.results.state = state

        self.results.kinematics = self._extract(self.field)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
            kirchhoff_stress=self._kirchhoff_stress,
        )

    def _vector(
        self, field=None, parallel=False, jit=False, items=None, args=(), kwargs={}
    ):

        self.results.stress = self._gradient(
            field, parallel=parallel, jit=jit, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.h(parallel=parallel, jit=jit)
        v = self.results.state.v()
        p = self.results.state.p

        values = [
            form.integrate(parallel=parallel, jit=jit)[0]
            + h * (self.bulk * (v / self.V - 1) - p)
        ]

        self.results.force = form.assemble(values=values)

        return self.results.force

    def _matrix(
        self, field=None, parallel=False, jit=False, items=None, args=(), kwargs={}
    ):

        self.results.elasticity = self._hessian(
            field, parallel=parallel, jit=jit, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.elasticity,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.h(parallel=parallel, jit=jit)

        values = [
            form.integrate(parallel=parallel, jit=jit)[0]
            + self.bulk / self.V * dya(h, h)
        ]

        self.results.stiffness = form.assemble(values=values)

        return self.results.stiffness

    def _extract(self, field, parallel=False, jit=False):

        u = field[0].values
        u0 = self.results.state.u
        h = self.results.state.h(parallel=parallel, jit=jit)
        v = self.results.state.v()
        J = self.results.state.J
        p = self.results.state.p

        du = (u - u0)[field.region.mesh.cells].transpose([1, 2, 0])

        # change of state variables due to change of displacement field
        dJ = ddot(h, du, n=1) / self.V + (v / self.V - J)
        dp = self.bulk * (dJ + J - 1) - p

        self.field = field
        self.results.kinematics = self.results.state.F = self.field.extract()

        # update state variables
        self.results.state.p = p + dp
        self.results.state.J = J + dJ
        self.results.state.u = u

        return self.results.kinematics

    def _gradient(self, field=None, parallel=False, jit=False, args=(), kwargs={}):

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel, jit=jit)

        dJdF = self._area_change.function
        F = self.results.kinematics[0]
        statevars = self.results.statevars

        p = self.results.state.p

        gradient = self.umat.gradient([F, statevars], *args, **kwargs)

        self.results.stress = [gradient[0] + p * dJdF([F])[0]]
        self.results._statevars = gradient[-1]

        return self.results.stress

    def _hessian(self, field=None, parallel=False, jit=False, args=(), kwargs={}):

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel, jit=jit)

        d2JdF2 = self._area_change.gradient
        F = self.results.kinematics[0]
        statevars = self.results.statevars
        p = self.results.state.p

        self.results.elasticity = [
            self.umat.hessian([F, statevars], *args, **kwargs)[0] + p * d2JdF2([F])[0]
        ]

        return self.results.elasticity

    def _kirchhoff_stress(self, field=None):

        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]

        return dot(P, transpose(F))

    def _cauchy_stress(self, field=None):

        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]
        J = det(F)

        return dot(P, transpose(F)) / J

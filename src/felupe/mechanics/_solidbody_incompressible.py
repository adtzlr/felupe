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

from .._assembly import IntegralForm
from .._field import FieldAxisymmetric
from ..constitution import AreaChange
from ..math import ddot, det, dot, dya, transpose
from ._helpers import Assemble, Evaluate, Results, StateNearlyIncompressible


class SolidBodyNearlyIncompressible:
    r"""A (nearly) incompressible SolidBody with methods for the assembly of sparse
    vectors/matrices for a material with optional state variables.

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

    Notes
    -----
    The volumetric material behaviour is hard-coded and is defined by the strain energy
    function.

    ..  math::

        U(J) = \frac{K}{2} (J - 1)^2


    **Hu-Washizu Three-Field-Variation Principle**

    The Three-Field-Variation :math:`(\boldsymbol{u},p,J)` leads to a linearized 
    equation system with nine sub block-matrices. Due to the fact that the equation 
    system is derived by a potential, the matrix is symmetric and hence, only six 
    independent sub-matrices have to be evaluated. Furthermore, by the application of 
    the mean dilatation technique, two of the remaining six sub-matrices are identified
    to be zero. That means four sub-matrices are left to be evaluated, where two 
    non-zero sub-matrices are scalar-valued entries.

    ..  math::

        \begin{bmatrix} 
            \boldsymbol{A}   & \boldsymbol{b} & \boldsymbol{0} \\
            \boldsymbol{b}^T &             0  &            -c  \\
            \boldsymbol{0}^T &            -c  &             d
        \end{bmatrix} \cdot \begin{bmatrix} 
            \boldsymbol{x} \\
                        y  \\
                        z 
        \end{bmatrix} = \begin{bmatrix} 
            \boldsymbol{u} \\
                        v  \\
                        w 
        \end{bmatrix}

    An alternative representation of the equation system, only dependent on the primary
    unknowns :math:`\boldsymbol{u}` is carried out. To do so, the second line is 
    multiplied by :math:`\frac{d}{c}`.

    ..  math::

        \begin{bmatrix} 
                        \boldsymbol{A}   & \boldsymbol{b} & \boldsymbol{0} \\
            \frac{d}{c}~\boldsymbol{b}^T &             0  &            -d  \\
                        \boldsymbol{0}^T &            -c  &             d
        \end{bmatrix} \cdot \begin{bmatrix} 
            \boldsymbol{x} \\
                        y  \\
                        z 
        \end{bmatrix} = \begin{bmatrix} 
            \boldsymbol{u} \\
            \frac{d}{c}~v  \\
            -w 
        \end{bmatrix}

    Now, equations two and three are summed up. This eliminates one of the three 
    unknowns.

    ..  math::

        \begin{bmatrix} 
                        \boldsymbol{A}   & \boldsymbol{b} \\
            \frac{d}{c}~\boldsymbol{b}^T &    -c
        \end{bmatrix} \cdot \begin{bmatrix} 
            \boldsymbol{x} \\
                y
        \end{bmatrix} = \begin{bmatrix} 
            \boldsymbol{u} \\
            \frac{d}{c}~v + w
        \end{bmatrix}

    Next, the second equation is left-multiplied by :math:`\frac{1}{c}~\boldsymbol{b}` 
    and both equations are summed up again.

    ..  math::

        \begin{bmatrix} 
            \boldsymbol{A} + \frac{d}{c^2}~\boldsymbol{b} \otimes \boldsymbol{b}
        \end{bmatrix} \cdot \begin{bmatrix} 
            \boldsymbol{x}
        \end{bmatrix} = \begin{bmatrix} 
            \boldsymbol{u} + \frac{d}{c^2}~\boldsymbol{b}~v + 
                \frac{1}{c}~\boldsymbol{b}~w
        \end{bmatrix}

    The secondary unknowns are evaluated after solving the primary unknowns.

    ..  math::

        z &= \frac{1}{c}~\boldsymbol{b}^T \boldsymbol{x} - \frac{1}{c}~v

        y &= \frac{d}{c}~z - \frac{1}{c}~w

    For the mean-dilatation technique, the variables, equations as well as sub-matrices
    are evaluated. Note that the pairs of indices :math:`(ai)` and :math:`(bk)` have to
    be treated as 1d-vectors.

    ..  math::

        A_{aibk} &= \int_V \frac{\partial h_a}{\partial X_J}  \left( 
            \frac{\partial^2 \overset{\wedge}{\psi}}{\partial F_{iJ} \partial F_{kL}} + 
            p \frac{\partial^2 J}{\partial F_{iJ} \partial F_{kL}} \right) 
            \frac{\partial h_b}{\partial X_L} \ dV

        b_{ai} &= \int_V \frac{\partial h_a}{\partial X_J} 
            \frac{\partial J}{\partial F_{iJ}} \ dV

        c &= \int_V \ dV = V

        d &= \int_V \frac{\partial^2 U(\bar{J})}{\partial \bar{J} \partial \bar{J}} \ dV
           = \bar{U}'' V

    and

    ..  math::

        x_{ai} &= \delta {u}_{ai}

        y &= \delta p

        z &= \delta \bar{J}

    as well as

    ..  math::

        u_{ai} (= -r_{ai}) &= -\int_V \frac{\partial h_a}{\partial X_J} \left( 
            \frac{\partial \overset{\wedge}{\psi}}{\partial F_{iJ}} + 
            p \frac{\partial J}{\partial F_{iJ}} \right) \ dV

        v &= -\int_V (J - \bar{J}) \ dV = \bar{J} V - v

        z &= -\int_V (\bar{U}' - p) \ dV = p V - \bar{U}' V

    """

    def __init__(self, umat, field, bulk, state=None, statevars=None):
        self.umat = umat
        self.field = field
        self.bulk = bulk

        self._area_change = AreaChange()
        self._form = IntegralForm

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

    def _vector(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        self.results.stress = self._gradient(
            field, parallel=parallel, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.h(parallel=parallel)
        v = self.results.state.v()
        p = self.results.state.p

        values = [
            form.integrate(parallel=parallel)[0]
            + h * (self.bulk * (v / self.V - 1) - p)
        ]

        self.results.force = form.assemble(values=values)

        return self.results.force

    def _matrix(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        self.results.elasticity = self._hessian(
            field, parallel=parallel, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.elasticity,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.h(parallel=parallel)

        values = [form.integrate(parallel=parallel)[0] + self.bulk / self.V * dya(h, h)]

        self.results.stiffness = form.assemble(values=values)

        return self.results.stiffness

    def _extract(self, field, parallel=False):
        u = field[0].values
        u0 = self.results.state.u
        h = self.results.state.h(parallel=parallel)
        v = self.results.state.v()
        J = self.results.state.J
        p = self.results.state.p

        du = (u - u0)[field.region.mesh.cells].transpose([1, 2, 0])

        # change of state variables due to change of displacement field
        dJ = ddot(h, du, mode=(2, 2)) / self.V + (v / self.V - J)
        dp = self.bulk * (dJ + J - 1) - p

        self.field = field
        self.results.kinematics = self.results.state.F = self.field.extract()

        # update state variables
        self.results.state.p = p + dp
        self.results.state.J = J + dJ
        self.results.state.u = u

        return self.results.kinematics

    def _gradient(self, field=None, parallel=False, args=(), kwargs={}):
        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

        dJdF = self._area_change.function
        F = self.results.kinematics[0]
        statevars = self.results.statevars

        p = self.results.state.p

        gradient = self.umat.gradient([F, statevars], *args, **kwargs)

        self.results.stress = [gradient[0] + p * dJdF([F])[0]]
        self.results._statevars = gradient[-1]

        return self.results.stress

    def _hessian(self, field=None, parallel=False, args=(), kwargs={}):
        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

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

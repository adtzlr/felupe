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
from ..constitution import AreaChange, ThreeFieldVariation
from ..math import det, dot, ddot, inv, transpose
from ..field import FieldDual, FieldContainer
from ._solidbody import Solid
from ._helpers import Assemble, Evaluate, Results
import numpy as np
from types import SimpleNamespace


class SolidBodyNearlyIncompressibleX(Solid):
    r"""A SolidBody with methods for the assembly of sparse vectors/matrices.

    Parameters
    ----------
    umat : class
        A class which provides methods for evaluating the gradient and the hessian of
        the strain energy density function per unit undeformed volume. The function
        signatures must be ``dψdF, ζ_new = umat.gradient([F, ζ])`` for
        the gradient and ``d2ψdFdF = umat.hessian([F, ζ])`` for the hessian of
        the strain energy density function :math:`\psi(\boldsymbol{F})`, where
        :math:`\boldsymbol{F}` is the deformation gradient tensor and :math:`\zeta`
        holds the array of internal state variables.
    field : FieldContainer
        A field container with one or more fields.
    statevars : ndarray or None, optional
        Array of initial internal state variables (default is None).

    """

    def __init__(self, umat, field, statevars=None):
        self.umat = ThreeFieldVariation(umat)
        self.field = field
        self.results = Results(stress=True, elasticity=True)

        if statevars is not None:
            self.results.statevars = statevars
        else:
            statevars_shape = (0,)
            if hasattr(umat, "x"):
                statevars_shape = umat.x[-1].shape
            self.results.statevars = np.zeros(
                (
                    *statevars_shape,
                    field.region.quadrature.npoints,
                    field.region.mesh.ncells,
                )
            )

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
            kirchhoff_stress=self._kirchhoff_stress,
        )

        self.results.state = SimpleNamespace()
        displacement = self.field[0]
        pressure = FieldDual(self.field.region)
        volume_ratio = FieldDual(self.field.region, values=1)
        self.results.state.field = type(field)([displacement, pressure, volume_ratio])
        self.results.state.u = displacement.interpolate()
        self.results.state.p = pressure.interpolate()
        self.results.state.J = volume_ratio.interpolate()
        self.results.kinematics = self._extract(self.results.state.field)

        self._area_change = AreaChange()

    def _vector(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        if field is not None:
            self.field = field

        self.results.stress = self._gradient(field, args=args, kwargs=kwargs)
        form = IntegralForm(
            fun=self.results.stress[slice(items)],
            v=self.results.state.field,
            dV=self.results.state.field.region.dV,
        )

        self.results.force_values = form.integrate(
            parallel=parallel, out=self.results.force_values
        )

        [fu, fp, fJ] = self.results.force_values

        fu = fu[:, :, None, ...]
        fp = fp[:, None, ...]
        fJ = fJ[:, None, ...]

        self.results.stiffness = self._matrix()
        [Kuu, Kup, KuJ, Kpp, KpJ, KJJ] = self.results.stiffness_values

        Kuu = Kuu[:, :, :, :, None, ...]
        Kup = Kup[:, :, None, None, ...]
        KuJ = KuJ[:, :, None, None, ...]
        Kpp = Kpp[:, None, :, None, ...]
        KpJ = KpJ[:, None, :, None, ...]
        KJJ = KJJ[:, None, :, None, ...]

        ddot44 = lambda a, b: ddot(a, b, mode=(4, 4))
        ddot42 = lambda a, b: ddot(a, b, mode=(4, 2))

        iKpJ = inv(KpJ)
        iKJp = transpose(iKpJ)
        f = (
            fu
            + ddot42(ddot44(ddot44(ddot44(Kup, iKpJ), KJJ), iKJp), fp)
            - ddot42(ddot44(Kup, iKpJ), fJ)
        )

        form_u = IntegralForm(
            fun=self.results.stress[:1],
            v=FieldContainer(self.field[:1]),
            dV=self.field.region.dV,
        )
        self.results.force = form_u.assemble(values=[f])

        return self.results.force

    def _matrix(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        if field is not None:
            self.field = field
            self.results.state.field.link(self.field)

        self.results.elasticity = self._hessian(field, args=args, kwargs=kwargs)

        form = IntegralForm(
            fun=self.results.elasticity[slice(items)],
            v=self.results.state.field,
            u=self.results.state.field,
            dV=self.results.state.field.region.dV,
        )

        self.results.stiffness_values = form.integrate(
            parallel=parallel, out=self.results.stiffness_values
        )

        [Kuu, Kup, KuJ, Kpp, KpJ, KJJ] = self.results.stiffness_values

        Kuu = Kuu[:, :, :, :, None, ...]
        Kup = Kup[:, :, None, None, ...]
        KuJ = KuJ[:, :, None, None, ...]
        Kpp = Kpp[:, None, :, None, ...]
        KpJ = KpJ[:, None, :, None, ...]
        KJJ = KJJ[:, None, :, None, ...]

        ddot44 = lambda a, b: ddot(a, b, mode=(4, 4))

        iKpJ = inv(KpJ)
        iKJp = transpose(iKpJ)
        Kpu = transpose(Kup)

        K = Kuu + ddot44(ddot44(ddot44(ddot44(Kup, iKpJ), KJJ), iKJp), Kpu)

        form_u = IntegralForm(
            fun=self.results.elasticity[:1],
            v=FieldContainer(self.field[:1]),
            u=FieldContainer(self.field[:1]),
            dV=self.field.region.dV,
        )
        self.results.stiffness = form_u.assemble(values=[K])

        return self.results.stiffness

    def _extract(self, field):
        self.field = field
        self.results.state.field.link(self.field)
        self.results.kinematics = self.results.state.field.extract(
            out=self.results.kinematics
        )

        self._vector()
        self._matrix()

        [fu, fp, fJ] = self.results.force_values
        [Kuu, Kup, KuJ, Kpp, KpJ, KJJ] = self.results.stiffness_values

        fp = fp[:, None, ...]
        fJ = fJ[:, None, ...]

        Kuu = Kuu[:, :, :, :, None, ...]
        Kup = Kup[:, :, None, None, ...]
        KuJ = KuJ[:, :, None, None, ...]
        Kpp = Kpp[:, None, :, None, ...]
        KpJ = KpJ[:, None, :, None, ...]
        KJJ = KJJ[:, None, :, None, ...]

        iKpJ = inv(KpJ)
        iKJp = transpose(iKpJ)
        Kpu = transpose(Kup)

        ddot44 = lambda a, b: ddot(a, b, mode=(4, 4))
        ddot42 = lambda a, b: ddot(a, b, mode=(4, 2))

        du = field[0].interpolate() - self.results.state.u
        dJ = ddot42(iKJp, fp) - ddot42(ddot44(iKJp, Kpu), du)
        dp = ddot42(iKpJ, fJ) - ddot42(ddot44(iKpJ, KJJ), dJ)

        print(dJ.shape)

        # update state variables
        self.results.state.p += dp
        self.results.state.J += dJ
        self.results.state.u = field[0].interpolate()

        return self.results.kinematics

    def _gradient(self, field=None, args=(), kwargs={}):
        if field is not None:
            self.field = field
            self.results.state.field.link(self.field)
            self.results.kinematics = self._extract(self.results.state.field)

        gradient = self.umat.gradient(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )

        self.results.stress, self.results._statevars = gradient[:-1], gradient[-1]

        return self.results.stress

    def _hessian(self, field=None, args=(), kwargs={}):
        if field is not None:
            self.field = field
            self.results.state.field.link(self.field)
            self.results.kinematics = self._extract(self.results.state.field)

        self.results.elasticity = self.umat.hessian(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )

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

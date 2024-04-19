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
import warnings

import numpy as np

from ..assembly import IntegralForm
from ..constitution import AreaChange
from ..field import FieldAxisymmetric
from ..math import ddot, det, dot, dya, transpose
from ._helpers import Assemble, Evaluate, Results, StateNearlyIncompressible
from ._solidbody import Solid


class SolidBodyNearlyIncompressible(Solid):
    r"""A (nearly) incompressible solid body with methods for the assembly of sparse
    vectors/matrices. The constitutive material definition must provide the distortional
    part of the strain energy density function per unit undeformed volume only. The
    volumetric part of the strain energy density function is automatically added on
    additonal internal (dual) cell-wise constant fields.

    Parameters
    ----------
    umat : class
        A class which provides methods for evaluating the gradient and the hessian of
        the isochoric part of the strain energy density function per unit undeformed
        volume :math:`\hat\psi(\boldsymbol{F})`. The function signatures must be
        ``dψdF, ζ_new = umat.gradient([F, ζ])`` for the gradient and
        ``d2ψdFdF = umat.hessian([F, ζ])`` for the hessian of
        the strain energy density function :math:`\hat{\psi}(\boldsymbol{F})`, where
        :math:`\boldsymbol{F}` is the deformation gradient tensor and :math:`\zeta`
        holds the array of internal state variables.
    field : FieldContainer
        A field container with one or more fields.
    bulk : float
        The bulk modulus of the volumetric part of the strain energy function
        :math:`U(\bar{J})=K(\bar{J}-1)^2/2`.
    state : StateNearlyIncompressible or None, optional
        A valid initial state for a (nearly) incompressible solid (default is None).
    statevars : ndarray or None, optional
        Array of initial internal state variables (default is None).

    Notes
    -----
    The total potential energy of internal forces for a three-field variational
    approach, suitable for nearly-incompressible material behaviour, is given in Eq.
    :eq:`nearlyincompressible`

    ..  math::
        :label: nearlyincompressible

        \Pi_{int}(\boldsymbol{F}, p, \bar{J}) =
            \int_V \hat{\psi}(\boldsymbol{F})\ dV +
            \int_V U(\bar{J})\ dV +
            \int_V p (J - \bar{J})\ dV
    
    with its variations, see Eq. :eq:`nearlyinc-variations`
    
    ..  math::
        :label: nearlyinc-variations

        \delta_\boldsymbol{u}(\Pi_{int}) &=
            \int_V \left( \frac{\partial \hat{\psi}}{\partial \boldsymbol{F}} +
            p\ \frac{\partial J}{\partial \boldsymbol{F}} \right) : 
            \delta\boldsymbol{F}\ dV
        \longrightarrow \boldsymbol{r}_\boldsymbol{u}

        \delta_p(\Pi_{int}) &=
            \int_V \left( J - \bar{J} \right)\ \delta p\ dV
        \longrightarrow r_p

        \delta_\bar{J}(\Pi_{int}) &=
            \int_V \left( K \left( \bar{J} - 1 \right) - p \right)\ \delta \bar{J}\ dV
        \longrightarrow r_{\bar{J}}
    
    and linearizations, see Eq. :eq:`nearlyinc-linearizations` [1]_ [2]_ [3]_. The
    right-arrows in Eq. :eq:`nearlyinc-variations` and Eq.
    :eq:`nearlyinc-linearizations` represent the assembly into system scalars, vectors
    or matrices.
    
    ..  math::
        :label: nearlyinc-linearizations

        \delta_\boldsymbol{u}\Delta_\boldsymbol{u}(\Pi_{int}) &=
            \int_V \delta\boldsymbol{F} : \left(
                \frac{\partial^2 \hat{\psi}}{
                    \partial \boldsymbol{F}\ \partial \boldsymbol{F}
                } +
                p \frac{\partial^2 J}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}
            \right) : \Delta\boldsymbol{F}\ dV
        \longrightarrow \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}}

        
        \delta_\boldsymbol{u}\Delta_p(\Pi_{int}) &= \int_V \delta \boldsymbol{F} : 
            \frac{\partial J}{\partial \boldsymbol{F}}\ \Delta p\ dV
        \longrightarrow \boldsymbol{K}_{\boldsymbol{u}p}
        
        \delta_p\Delta_{\bar{J}}(\Pi_{int}) &= 
            \int_V \delta p\ (-1)\ \Delta \bar{J}\ dV 
        \longrightarrow -V
        
        \delta_{\bar{J}}\Delta_{\bar{J}}(\Pi_{int}) &= 
            \int_V \delta \bar{J}\ K\ \Delta \bar{J}\ dV 
        \longrightarrow K\ V
    
    The assembled constraint equations for the variations w.r.t. the dual fields
    :math:`p` and :math:`\bar{J}` are given in Eq. :eq:`nearlyinc-constraints`.
    
    ..  math::
        :label: nearlyinc-constraints

        r_p &= \left( \frac{v}{V} - \bar{J} \right) V

        r_{\bar{J}} &= \left( K (\bar{J} - 1) - p \right) V


    The volumetric part of the strain energy density function is denoted in Eq.
    :eq:`nearlyincompressible-volumetric` along with its first and second derivatives.

    ..  math::
        :label: nearlyincompressible-volumetric

        \bar{U} &= \frac{K}{2} \left( \bar{J} - 1 \right)^2

        \bar{U}' &= K \left( \bar{J} - 1 \right)

        \bar{U}'' &= K


    **Hu-Washizu Three-Field-Variational Principle**

    The Three-Field-Variation :math:`(\boldsymbol{u},p,\bar{J})` leads to a linearized
    equation system with nine sub block-matrices, see Eq. :eq:`hu-washizu` [4]_.
    Due to the fact that the equation system is derived by a potential, the matrix is
    symmetric and hence, only six independent sub-matrices have to be evaluated.
    Furthermore, by the application of the mean dilatation technique, two of the
    remaining six sub-matrices are identified to be zero. That means four sub-matrices
    are left to be evaluated, where two non-zero sub-matrices are scalar-valued entries.

    ..  math::
        :label: hu-washizu

        \begin{bmatrix}
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}} 
                & \boldsymbol{K}_{\boldsymbol{u}p} & \boldsymbol{0} \\
            \boldsymbol{K}_{\boldsymbol{u}p}^T & 0 & -V \\
            \boldsymbol{0}^T & -V  & K\ V
        \end{bmatrix} \cdot \begin{bmatrix}
            \delta \boldsymbol{u} \\
            \delta p \\
            \delta \bar{J}
        \end{bmatrix} + \begin{bmatrix}
            \boldsymbol{r}_\boldsymbol{u} \\
            r_p \\
            r_\bar{J}
        \end{bmatrix} = 
        \begin{bmatrix}
            \boldsymbol{0}\\
            0 \\
            0
        \end{bmatrix}

    A condensed representation of the equation system, only dependent on the primary
    unknowns :math:`\boldsymbol{u}` is carried out. To do so, the second line is
    multiplied by the bulk modulus :math:`K`, see Eq. :eq:`nearlyinc-condensed`.

    ..  math::
        :label: nearlyinc-condensed

        \begin{bmatrix}
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}} 
                & \boldsymbol{K}_{\boldsymbol{u}p} & \boldsymbol{0} \\
            K \boldsymbol{K}_{\boldsymbol{u}p}^T & 0 & -K\ V \\
            \boldsymbol{0}^T & -V  & K\ V
        \end{bmatrix} \cdot \begin{bmatrix}
            \delta \boldsymbol{u} \\
            \delta p \\
            \delta \bar{J}
        \end{bmatrix} + \begin{bmatrix}
            \boldsymbol{r}_\boldsymbol{u} \\
            K\ r_p \\
            r_\bar{J}
        \end{bmatrix} = 
        \begin{bmatrix}
            \boldsymbol{0}\\
            0 \\
            0
        \end{bmatrix}

    Lines two and three are contracted by summation as given in Eq.
    :eq:`nearlyinc-contracted-sum`. This eliminates :math:`\bar{J}` from the unknowns.

    ..  math::
        :label: nearlyinc-contracted-sum

        \begin{bmatrix}
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}} 
                & \boldsymbol{K}_{\boldsymbol{u}p} \\
            K \boldsymbol{K}_{\boldsymbol{u}p}^T & -V
        \end{bmatrix} \cdot \begin{bmatrix}
            \delta \boldsymbol{u} \\
            \delta p
        \end{bmatrix} + \begin{bmatrix}
            \boldsymbol{r}_\boldsymbol{u} \\
            K\ r_p + r_\bar{J}
        \end{bmatrix} = 
        \begin{bmatrix}
            \boldsymbol{0}\\
            0
        \end{bmatrix}

    Next, the second line is left-expanded by 
    :math:`\frac{1}{V}~\boldsymbol{K}_{\boldsymbol{u}p}` and both equations are summed
    up again, see :eq:`nearlyinc-contracted-sum-final`.

    ..  math::
        :label: nearlyinc-contracted-sum-final

        \left(
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}} 
                + \frac{K}{V}~\boldsymbol{K}_{\boldsymbol{u}p} \otimes 
                    \boldsymbol{K}_{\boldsymbol{u}p}
        \right) \cdot \delta \boldsymbol{u} +
        \boldsymbol{r}_\boldsymbol{u} + \frac{K~r_p + r_\bar{J}}{V}
            \boldsymbol{K}_{\boldsymbol{u}p} = \boldsymbol{0}

    The secondary unknowns are evaluated after solving the primary unknowns, see
    Eq. :eq:`nearlyinc-final`.

    ..  math::
        :label: nearlyinc-final

        \delta \bar{J} &= \frac{1}{V} \delta \boldsymbol{u} \cdot 
            \boldsymbol{K}_{\boldsymbol{u}p} + \frac{v}{V} - \bar{J}

        \delta p &= K \left(\bar{J} + \delta \bar{J} - 1 \right) - p

    The condensed constraint equation in Eq. :eq:`nearlyinc-contracted-sum-final` is
    given in Eq. :eq:`nearlyinc-constraint`
    
    ..  math::
        :label: nearlyinc-constraint
        
        \frac{K~r_p + r_{\bar{J}}}{V} = K \left( \frac{v}{V} - 1 \right) - p
    
    and the deformed volume is evaluated by Eq. :eq:`nearlyinc-deformed-volume`.
    
    ..  math::
        :label: nearlyinc-deformed-volume
        
        v = \int_V J\ dV


    Examples
    --------
    ..  pyvista-plot::
        
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
        >>>
        >>> umat = fem.NeoHooke(mu=1)
        >>> solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
        >>>
        >>> table = fem.math.linsteps([0, 1], num=5)
        >>> step = fem.Step(
        ...     items=[solid],
        ...     ramp={boundaries["move"]: table},
        ...     boundaries=boundaries,
        ... )
        >>>
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> solid.plot("Principal Values of Cauchy Stress").show()
    
    References
    ----------
    ..  [1] J. Bonet, A. J. Gil, and R. D. Wood, "Nonlinear Solid Mechanics for Finite
        Element Analysis: Statics". Cambridge University Press, Jun. 05, 2016. 
        doi: 10.1017/cbo9781316336144.

    ..  [2] O. C. Zienkiewicz, R. L. Taylor and J.Z. Zhu, "The Finite Element Method:
        its Basis and Fundamentals". Elsevier, 2013. doi: 10.1016/c2009-0-24909-9.
    
    ..  [3] G. A. Holzapfel, "Nonlinear Solid Mechanics. A Continuum Approach for
        Engineering". Wiley, 2000. isbn: 047182304X.
    
    ..  [4] J. A. Schönherr, P. Schneider, and C. Mittelstedt, "Robust hybrid/mixed
        finite elements for rubber-like materials under severe compression",
        Computational Mechanics, vol. 70, no. 1. Springer Science and Business Media
        LLC, pp. 101–122, Mar. 31, 2022. doi: 10.1007/s00466-022-02157-y.
    
    See Also
    --------
    felupe.StateNearlyIncompressible : A State with internal cell-wise constant fields
        for (nearly) incompressible solid bodies.
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

    def _vector(self, field=None, parallel=False, items=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        self.results.stress = self._gradient(
            field, parallel=parallel, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.integrate_shape_function_gradient(
            parallel=parallel, out=self.results._force_values
        )
        v = self.results.state.volume()
        p = self.results.state.p

        constraint = np.multiply(h, self.bulk * (v / self.V - 1) - p, out=h)

        self.results.force_values = form.integrate(
            parallel=parallel, out=self.results.force_values
        )
        np.add(
            self.results.force_values[0], constraint, out=self.results.force_values[0]
        )

        self.results.force = form.assemble(values=self.results.force_values)

        return self.results.force

    def _matrix(self, field=None, parallel=False, items=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        self.results.elasticity = self._hessian(
            field, parallel=parallel, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.elasticity,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.integrate_shape_function_gradient(
            parallel=parallel, out=self.results._force_values
        )
        H = dya(h, h, out=self.results._stiffness_values)
        bulk_H = np.multiply(H, self.bulk, out=H)
        constraint = np.divide(bulk_H, self.V, out=bulk_H)

        self.results.stiffness_values = form.integrate(
            parallel=parallel, out=self.results.stiffness_values
        )
        np.add(
            self.results.stiffness_values[0],
            constraint,
            out=self.results.stiffness_values[0],
        )

        self.results.stiffness = form.assemble(values=self.results.stiffness_values)

        return self.results.stiffness

    def _extract(self, field, parallel=False):
        u = field[0].values
        u0 = self.results.state.u
        h = self.results.state.integrate_shape_function_gradient(
            parallel=parallel, out=self.results._force_values
        )
        v = self.results.state.volume()

        du = (u - u0)[field.region.mesh.cells].transpose([1, 2, 0])

        self.field = field
        self.results.kinematics = self.results.state.F = self.field.extract(
            out=self.results.kinematics
        )

        # update state variables
        self.results.state.J[:] = (ddot(du, h, mode=(2, 2)) + v) / self.V
        self.results.state.p[:] = self.bulk * (self.results.state.J - 1)
        self.results.state.u[:] = u

        return self.results.kinematics

    def _gradient(self, field=None, parallel=False, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

        dJdF = self._area_change.function
        F = self.results.kinematics[0]
        statevars = self.results.statevars

        p = self.results.state.p

        if "out" in inspect.signature(self.umat.gradient).parameters:
            kwargs["out"] = self.results.gradient

        [self.results.gradient, self.results._statevars] = self.umat.gradient(
            [F, statevars], *args, **kwargs
        )
        self.results.stress = [
            np.add(self.results.gradient, p * dJdF([F])[0], out=self.results.gradient)
        ]

        return self.results.stress

    def _hessian(self, field=None, parallel=False, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

        d2JdF2 = self._area_change.gradient
        F = self.results.kinematics[0]
        statevars = self.results.statevars
        p = self.results.state.p

        if "out" in inspect.signature(self.umat.hessian).parameters:
            kwargs["out"] = self.results.hessian

        self.results.hessian = self.umat.hessian([F, statevars], *args, **kwargs)[0]
        self.results.elasticity = [
            np.add(self.results.hessian, p * d2JdF2([F])[0], out=self.results.hessian)
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

        if P.shape[:2] == (2, 2):
            warnings.warn(
                "\n".join(
                    [
                        "Cauchy stress tensor can't be evaluated on a 2d-Field.",
                        "Falling-back to the Kirchhoff stress tensor.",
                    ]
                )
            )
            J = 1
        else:
            J = det(F)

        return dot(P, transpose(F)) / J

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
import tensortrax as tr

from ...math import cdya_ik, dot, transpose
from .._material import Material


class Hyperelastic(Material):
    r"""A hyperelastic material definition with a given function for the strain energy
    density function per unit undeformed volume with Automatic Differentiation provided
    by :mod:`tensortrax`.

    Parameters
    ----------
    fun : callable
        A strain energy density function in terms of the right Cauchy-Green deformation
        tensor :math:`\boldsymbol{C}`. Function signature must be
        ``fun = lambda C, **kwargs: psi`` for functions without state variables and
        ``fun = lambda C, statevars, **kwargs: [psi, statevars_new]`` for functions
        with state variables. The right Cauchy-Green deformation tensor will be a
        :class:`tensortrax.Tensor` when the function is evaluated. It is important to
        use only differentiable math-functions from :mod:`tensortrax.math`.
    nstatevars : int, optional
        Number of state variables (default is 0).
    parallel : bool, optional
        A flag to invoke threaded strain energy density function evaluations (default
        is False). May introduce additional overhead for small-sized problems.
    **kwargs : dict, optional
        Optional keyword-arguments for the strain energy density function.

    Notes
    -----
    The strain energy density function :math:`\psi` must be given in terms of the right
    Cauchy-Green deformation tensor
    :math:`\boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}`.

    ..  warning::
        It is important to use only differentiable math-functions from
        :mod:`tensortrax.math`!

    Take this minimal code-block as template

    ..  math::

        \psi = \psi(\boldsymbol{C})

    ..  code-block::

        import felupe as fem
        import felupe.constitution.tensortrax as mat
        import tensortrax.math as tm

        def neo_hooke(C, mu):
            "Strain energy function of the Neo-Hookean material formulation."
            return mu / 2 * (tm.linalg.det(C) ** (-1/3) * tm.trace(C) - 3)

        umat = mat.Hyperelastic(neo_hooke, mu=1)

    and this code-block for material formulations with state variables.

    ..  math::

        \psi = \psi(\boldsymbol{C}, \boldsymbol{\zeta})

    ..  code-block::

        import felupe as fem
        import felupe.constitution.tensortrax as mat
        import tensortrax.math as tm

        def viscoelastic(C, Cin, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            Cu = tm.linalg.det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            Ci = tm.special.from_triu_1d(Cin, like=C) + mu / eta * dtime * Cu
            Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci

            # first invariant of elastic part of right Cauchy-Green deformation tensor
            I1 = tm.trace(Cu @ tm.linalg.inv(Ci))

            # strain energy function and state variable
            return mu / 2 * (I1 - 3), tm.special.triu_1d(Ci)

        umat = mat.Hyperelastic(viscoelastic, mu=1, eta=1, dtime=1, nstatevars=6)

    ..  note::
        See the `documentation of tensortrax <https://tensortrax.readthedocs.io>`_
        for further details.

    Examples
    --------
    View force-stretch curves on elementary incompressible deformations.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> import felupe.constitution.tensortrax as mat
        >>> import tensortrax.math as tm
        >>>
        >>> def neo_hooke(C, mu):
        ...     "Strain energy function of the Neo-Hookean material formulation."
        ...     return mu / 2 * (tm.linalg.det(C) ** (-1/3) * tm.trace(C) - 3)
        >>>
        >>> umat = mat.Hyperelastic(neo_hooke, mu=1)
        >>> ax = umat.plot(incompressible=True)

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    See Also
    --------
    felupe.constitution.tensortrax.models.hyperelastic.saint_venant_kirchhoff : Strain
        energy function of the Saint Venant-Kirchhoff material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.neo_hooke : Strain energy
        function of the Neo-Hookean material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.mooney_rivlin : Strain energy
        function of the Mooney-Rivlin material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.yeoh : "Strain energy function of
        the Yeoh material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.third_order_deformation : Strain
        energy function of the Third-Order-Deformation material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.ogden : Strain energy function of
        the Ogden material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.arruda_boyce : Strain energy
        function of the Arruda-Boyce material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.extended_tube : Strain energy
        function of the Extended-Tube material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.van_der_waals : Strain energy
        function of the Van der Waals material formulation.
    felupe.constitution.tensortrax.models.hyperelastic.finite_strain_viscoelastic :
        Finite strain viscoelastic material formulation.

    """

    def __init__(self, fun, nstatevars=0, parallel=False, **kwargs):
        if nstatevars > 0:
            # split the original function into two sub-functions
            self.fun = tr.take(fun, item=0)
            self.fun_statevars = tr.take(fun, item=1)
        else:
            self.fun = fun

        self.parallel = parallel

        keyword_args = kwargs
        if hasattr(fun, "kwargs"):
            keyword_args = {**fun.kwargs, **keyword_args}

        super().__init__(
            stress=self._stress,
            elasticity=self._elasticity,
            nstatevars=nstatevars,
            **keyword_args,
        )

    def _stress(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        C = dot(transpose(F), F)
        dWdC = tr.gradient(self.fun, wrt=0, ntrax=2, parallel=self.parallel, sym=True)(
            C, *statevars, **kwargs
        )
        if self.nstatevars > 0:
            statevars_new = tr.function(
                self.fun_statevars, wrt=0, ntrax=2, parallel=self.parallel
            )(C, *statevars, **kwargs)
        else:
            statevars_new = None
        return [dot(F, 2 * dWdC), statevars_new]

    def _elasticity(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        C = dot(transpose(F), F)
        d2WdCdC, dWdC, W = tr.hessian(
            self.fun, wrt=0, ntrax=2, full_output=True, parallel=self.parallel, sym=True
        )(C, *statevars, **kwargs)
        A = 4 * np.einsum(
            "iI...,kK...,IJKL...->iJkL...", F, F, np.ascontiguousarray(d2WdCdC)
        )
        B = cdya_ik(np.eye(3), 2 * dWdC)
        return [np.sum(np.broadcast_arrays(A, B), axis=0)]

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

from .._material import Material as MaterialDefault


class Material(MaterialDefault):
    r"""A material definition with a given function for the partial derivative of the
    strain energy function w.r.t. the deformation gradient tensor with Automatic
    Differentiation provided by `tensortrax <https://github.com/adtzlr/tensortrax>`_.

    Parameters
    ----------
    fun : callable
        A gradient of the strain energy density function w.r.t. the deformation gradient
        tensor :math:`\boldsymbol{F}`. Function signature must be
        ``fun = lambda F, **kwargs: P`` for functions without state variables and
        ``fun = lambda F, statevars, **kwargs: [P, statevars_new]`` for functions
        with state variables. The deformation gradient tensor will be a
        :class:`tensortrax.Tensor` when the function is evaluated. It is important to
        use only differentiable math-functions from ``tensortrax.math``!
    nstatevars : int, optional
        Number of state variables (default is 0).
    parallel : bool, optional
        A flag to invoke threaded gradient of strain energy density function evaluations
        (default is False). May introduce additional overhead for small-sized problems.
    **kwargs : dict, optional
        Optional keyword-arguments for the gradient of the strain energy density
        function.

    Notes
    -----
    The gradient of the strain energy density function
    :math:`\frac{\partial \psi}{\partial \boldsymbol{F}}` must be given in terms of the
    deformation gradient tensor :math:`\boldsymbol{F}`.

    ..  warning::
        It is important to use only differentiable math-functions from
        ``tensortrax.math``!

    ..  math::

        \boldsymbol{P} = \frac{\partial \psi(\boldsymbol{F}, \boldsymbol{\zeta})}{
            \partial \boldsymbol{F}}

    Take this code-block as template

    ..  code-block::

        import felupe as fem
        import felupe.constitution.tensortrax as mat
        import tensortrax.math as tm

        def neo_hooke(F, mu):
            "First Piola-Kirchhoff stress of the Neo-Hookean material formulation."

            C = tm.dot(tm.transpose(F), F)
            Cu = tm.linalg.det(C) ** (-1/3) * C

            return mu * F @ tm.special.dev(Cu) @ tm.linalg.inv(C)

        umat = mat.Material(neo_hooke, mu=1)

    and this code-block for material formulations with state variables:

    ..  code-block::

        import felupe as fem
        import felupe.constitution.tensortrax as mat
        import tensortrax.math as tm

        def viscoelastic(F, Cin, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            C = tm.dot(tm.transpose(F), F)
            Cu = tm.linalg.det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            Ci = tm.special.from_triu_1d(Cin, like=C) + mu / eta * dtime * Cu
            Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci

            # second Piola-Kirchhoff stress tensor
            S = mu * tm.special.dev(Cu @ tm.linalg.inv(Ci)) @ tm.linalg.inv(C)

            # first Piola-Kirchhoff stress tensor and state variable
            return F @ S, tm.special.triu_1d(Ci)

        umat = mat.Material(viscoelastic, mu=1, eta=1, dtime=1, nstatevars=6)

    ..  note::
        See the `documentation of tensortrax <https://github.com/adtzlr/tensortrax>`_
        for further details.

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> import felupe.constitution.tensortrax as mat
        >>> import tensortrax.math as tm
        >>>
        >>> def neo_hooke(F, mu):
        ...     C = tm.dot(tm.transpose(F), F)
        ...     S = mu * tm.special.dev(tm.linalg.det(C)**(-1/3) * C) @ tm.linalg.inv(C)
        ...     return F @ S
        >>>
        >>> umat = mat.Material(neo_hooke, mu=1)
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

    """

    def __init__(self, fun, nstatevars=0, parallel=False, **kwargs):
        if nstatevars > 0:
            # split the original function into two sub-functions
            self.fun = tr.take(fun, item=0)
            self.fun_statevars = tr.take(fun, item=1)
        else:
            self.fun = fun

        self.parallel = parallel

        super().__init__(
            stress=self._stress,
            elasticity=self._elasticity,
            nstatevars=nstatevars,
            **kwargs,
        )

    def _stress(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        dWdF = tr.function(self.fun, wrt=0, ntrax=2, parallel=self.parallel)(
            F, *statevars, **kwargs
        )
        if self.nstatevars > 0:
            statevars_new = tr.function(
                self.fun_statevars, wrt=0, ntrax=2, parallel=self.parallel
            )(F, *statevars, **kwargs)
        else:
            statevars_new = None
        return [dWdF, statevars_new]

    def _elasticity(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        d2WdFdF = tr.jacobian(self.fun, wrt=0, ntrax=2, parallel=self.parallel)(
            F, *statevars, **kwargs
        )
        return [d2WdFdF]

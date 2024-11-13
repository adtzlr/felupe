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

import warnings

import jax
import numpy as np

from .._material import Material
from ._helpers import as_total_lagrange, vmap2


class Hyperelastic(Material):
    r"""A hyperelastic material definition with a given function for the strain energy
    density function per unit undeformed volume with Automatic Differentiation provided
    by :mod:`jax`.

    Parameters
    ----------
    fun : callable
        A strain energy density function in terms of the right Cauchy-Green deformation
        tensor :math:`\boldsymbol{C}`. Function signature must be
        ``fun = lambda C, **kwargs: psi`` for functions without state variables and
        ``fun = lambda C, statevars, **kwargs: [psi, statevars_new]`` for functions
        with state variables. It is important to use only differentiable math-functions
        from :mod:`jax`.
    nstatevars : int, optional
        Number of state variables (default is 0).
    jit : bool, optional
        A flag to invoke just-in-time compilation (default is True).
    parallel : bool, optional
        A flag to invoke parallel strain energy density function evaluations (default
        is False). If True, the quadrature points are executed in parallel. The number
        of devices must be greater or equal the number of quadrature points per cell.
    **kwargs : dict, optional
        Optional keyword-arguments for the strain energy density function.

    Notes
    -----
    The strain energy density function :math:`\psi` must be given in terms of the right
    Cauchy-Green deformation tensor
    :math:`\boldsymbol{C} = \boldsymbol{F}^T \boldsymbol{F}`.

    ..  warning::
        It is important to use only differentiable math-functions from :mod:`jax`!

    Take this minimal code-block as template

    ..  math::

        \psi = \psi(\boldsymbol{C})

    ..  code-block::

        import felupe as fem
        import felupe.constitution.jax as mat
        import jax.numpy as jnp

        def neo_hooke(C, mu):
            "Strain energy function of the Neo-Hookean material formulation."
            return mu / 2 * (jnp.linalg.det(C) ** (-1/3) * jnp.trace(C) - 3)

        umat = mat.Hyperelastic(neo_hooke, mu=1)

    and this code-block for material formulations with state variables.

    ..  math::

        \psi = \psi(\boldsymbol{C}, \boldsymbol{\zeta})

    ..  code-block::

        import felupe as fem
        import felupe.constitution.jax as mat
        import jax.numpy as jnp

        def viscoelastic(C, Cin, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            Cu = jnp.linalg.det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            Ci = Cin.reshape(3, 3) + mu / eta * dtime * Cu
            Ci = jnp.linalg.det(Ci) ** (-1 / 3) * Ci

            # first invariant of elastic part of right Cauchy-Green deformation tensor
            I1 = jnp.trace(Cu @ jnp.linalg.inv(Ci))

            # strain energy function and state variable
            return mu / 2 * (I1 - 3), Ci.ravel()

        umat = mat.Hyperelastic(viscoelastic, mu=1, eta=1, dtime=1, nstatevars=9)

    ..  note::
        See the `documentation of JAX <https://jax.readthedocs.io>`_ for further
        details. JAX uses single-precision (32bit) data types by default. This requires
        to relax the tolerance of :func:`~felupe.newtonrhapson` to ``tol=1e-4``. If
        required, JAX may be enforced to use double-precision at startup with
        ``jax.config.update("jax_enable_x64", True)``.

    Examples
    --------
    View force-stretch curves on elementary incompressible deformations.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> import felupe.constitution.jax as mat
        >>> import jax.numpy as jnp
        >>>
        >>> def neo_hooke(C, mu):
        ...     "Strain energy function of the Neo-Hookean material formulation."
        ...     return mu / 2 * (jnp.linalg.det(C) ** (-1/3) * jnp.trace(C) - 3)
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

    """

    def __init__(self, fun, nstatevars=0, jit=True, parallel=False, **kwargs):
        has_aux = nstatevars > 0
        self.fun = as_total_lagrange(fun)

        keyword_args = kwargs
        if hasattr(fun, "kwargs"):
            keyword_args = {**fun.kwargs, **keyword_args}

        super().__init__(
            stress=self._stress,
            elasticity=self._elasticity,
            nstatevars=nstatevars,
            **keyword_args,
        )

        in_axes = out_axes_grad = [2, 3]
        if nstatevars > 0:
            in_axes = out_axes_grad = [(2, 1), (3, 2)]

        out_axes_hess = [4, 5]
        if nstatevars > 0:
            out_axes_hess = [(4, 1), (5, 2)]

        methods = [jax.vmap, jax.vmap]
        if parallel:
            methods[0] = jax.pmap  # apply on quadrature-points
            jit = False  # pmap uses jit

        self._grad = vmap2(
            jax.grad(self.fun, has_aux=has_aux),
            in_axes=in_axes,
            out_axes=out_axes_grad,
            methods=methods,
        )
        self._hess = vmap2(
            jax.jacfwd(jax.grad(self.fun, has_aux=has_aux), has_aux=has_aux),
            in_axes=in_axes,
            out_axes=out_axes_hess,
            methods=methods,
        )

        if jit:
            self._grad = jax.jit(self._grad)
            self._hess = jax.jit(self._hess)

    def _stress(self, x, **kwargs):
        if self.nstatevars > 0:
            statevars = x[1]

        F = x[0]
        if self.nstatevars > 0:
            dWdF, statevars_new = self._grad(F, statevars, **kwargs)
            statevars_new = np.array(statevars_new)
        else:
            dWdF = self._grad(F, **kwargs)
            statevars_new = None

        return [np.array(dWdF), statevars_new]

    def _elasticity(self, x, **kwargs):
        if self.nstatevars > 0:
            statevars = x[1]

        F = x[0]
        if self.nstatevars > 0:
            d2WdFdF, statevars_new = self._hess(F, statevars, **kwargs)
        else:
            d2WdFdF = self._hess(F, **kwargs)
        return [np.array(d2WdFdF)]

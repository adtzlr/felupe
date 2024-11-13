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

from .._material import Material as MaterialDefault
from ._helpers import vmap2


class Material(MaterialDefault):
    r"""A material definition with a given function for the partial derivative of the
    strain energy function w.r.t. the deformation gradient tensor with Automatic
    Differentiation provided by :mod:`jax`.

    Parameters
    ----------
    fun : callable
        A gradient of the strain energy density function w.r.t. the deformation gradient
        tensor :math:`\boldsymbol{F}`. Function signature must be
        ``fun = lambda F, **kwargs: P`` for functions without state variables and
        ``fun = lambda F, statevars, **kwargs: [P, statevars_new]`` for functions
        with state variables. It is important to use only differentiable math-functions
        from :mod:`jax`.
    nstatevars : int, optional
        Number of state variables (default is 0).
    jit : bool, optional
        A flag to invoke just-in-time compilation (default is True).
    parallel : bool, optional
        A flag to invoke parallel function evaluations (default is False). If True, the
        quadrature points are executed in parallel. The number of devices must be
        greater or equal the number of quadrature points per cell.
    jacobian : callable or None, optional
        A callable for the Jacobian. Default is None, where :func:`jax.jacobian` is
        used. This may be used to switch to forward-mode differentian
        :func:`jax.jacfwd`.
    **kwargs : dict, optional
        Optional keyword-arguments for the gradient of the strain energy density
        function.

    Notes
    -----
    The gradient of the strain energy density function
    :math:`\frac{\partial \psi}{\partial \boldsymbol{F}}` must be given in terms of the
    deformation gradient tensor :math:`\boldsymbol{F}`.

    ..  warning::
        It is important to use only differentiable math-functions from :mod:`jax`!

    Take this code-block as template

    ..  code-block::

        import felupe as fem
        import felupe.constitution.jax as mat
        import jax.numpy as jnp

        def neo_hooke(F, mu):
            "First Piola-Kirchhoff stress of the Neo-Hookean material formulation."

            C = F.T @ F
            Cu = jnp.linalg.det(C) ** (-1/3) * C
            dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)

            return mu * F @ dev(Cu) @ jnp.linalg.inv(C)

        umat = mat.Material(neo_hooke, mu=1)

    and this code-block for material formulations with state variables:

    ..  code-block::

        import felupe as fem
        import felupe.constitution.jax as mat
        import jax.numpy as jnp

        def viscoelastic(F, Cin, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            C = F.T @ F
            Cu = jnp.linalg.det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            from_triu = lambda C: C[jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])]
            Ci = from_triu(Cin) + mu / eta * dtime * Cu
            Ci = jnp.linalg.det(Ci) ** (-1 / 3) * Ci

            # second Piola-Kirchhoff stress tensor
            dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)
            S = mu * dev(Cu @ jnp.linalg.inv(Ci)) @ jnp.linalg.inv(C)

            # first Piola-Kirchhoff stress tensor and state variable
            i, j = jnp.triu_indices(3)
            to_triu = lambda C: C[i, j]
            return F @ S, to_triu(Ci)

        umat = mat.Material(viscoelastic, mu=1, eta=1, dtime=1, nstatevars=6)

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
        >>> def neo_hooke(F, mu):
        ...     "First Piola-Kirchhoff stress of the Neo-Hookean material formulation."
        ...
        ...     C = F.T @ F
        ...     Cu = jnp.linalg.det(C) ** (-1/3) * C
        ...     dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)
        ...
        ...     return mu * F @ dev(Cu) @ jnp.linalg.inv(C)
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

    def __init__(
        self, fun, nstatevars=0, jit=True, parallel=False, jacobian=None, **kwargs
    ):
        has_aux = nstatevars > 0
        self.fun = fun

        if jacobian is None:
            jacobian = jax.jacobian

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
            self.fun, in_axes=in_axes, out_axes=out_axes_grad, methods=methods
        )
        self._hess = vmap2(
            jacobian(self.fun, has_aux=has_aux),
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

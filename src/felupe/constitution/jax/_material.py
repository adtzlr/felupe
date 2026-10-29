# SPDX-License-Identifier: GPL-3.0-or-later

import jax
import numpy as np

from .._material import Material as MaterialDefault
from .._material import takes_time_increment
from ._helpers import time_increment_as_array, vmap2


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

        def viscoelastic(F, statevars, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            C = F.T @ F
            Cu = jnp.linalg.det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            from_triu = lambda C: C[jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])]
            Cin = jnp.eye(3) + from_triu(statevars)
            Ci = Cin + mu / eta * dtime * Cu
            Ci = jnp.linalg.det(Ci) ** (-1 / 3) * Ci

            # second Piola-Kirchhoff stress tensor
            dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)
            S = mu * jnp.linalg.inv(C) @ dev(Cu @ jnp.linalg.inv(Ci))

            # first Piola-Kirchhoff stress tensor and new state variables
            i, j = jnp.triu_indices(3)
            to_triu = lambda C: C[i, j]
            return F @ S, to_triu(Ci - jnp.eye(3))

        umat = mat.Material(viscoelastic, mu=1, eta=1, dtime=1, nstatevars=6)

    The state variables are the components of the upper triangle of the deviation of the
    inelastic right Cauchy-Green deformation tensor from the identity, i.e. the initial
    state variables are zero. The stress is evaluated at the updated state variables and
    the elasticity tensor, its Jacobian, includes the linearization of the updated state
    variables, i.e. it is the consistent linearization.

    **Time-dependent materials**: If the function has an argument ``dtime`` without a
    default value, which is not given as keyword-argument of the material, the material
    is time-dependent (the attribute ``time_dependent`` is True). Then, the time
    increment is an input of the material, like the deformation gradient, i.e. it is
    given by the keyword-argument ``dtime`` of :meth:`gradient` and :meth:`hessian`.
    It is passed as an array of data type float to the just-in-time compiled functions,
    i.e. they are compiled only once for all time increments. A time increment which is
    given as keyword-argument of the material, like ``dtime=1`` in the code-block above,
    is a constant material parameter.

    ..  code-block::

        umat = mat.Material(viscoelastic, mu=1, eta=1, nstatevars=6)
        assert umat.time_dependent

        P, statevars_new = umat.gradient([F, statevars], dtime=0.1)

    ..  note::
        See the `documentation of JAX <https://jax.readthedocs.io>`_ for further
        details. JAX uses single-precision (32bit) data types by default. This requires
        to relax the tolerance of :func:`~felupe.newtonraphson` to ``tol=1e-4``. If
        required, JAX may be enforced to use double-precision at startup with
        ``jax.config.update("jax_enable_x64", True)``.

    Examples
    --------
    View force-stretch curves on elementary incompressible deformations.

    ..  plot::

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

        # the time increment is an input, if it is an argument of the (original)
        # function which is not given as material parameter
        self.time_dependent = takes_time_increment(fun, self.kwargs)

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
        if self.time_dependent:
            kwargs = time_increment_as_array(kwargs)

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
        if self.time_dependent:
            kwargs = time_increment_as_array(kwargs)

        if self.nstatevars > 0:
            statevars = x[1]

        F = x[0]
        if self.nstatevars > 0:
            d2WdFdF, statevars_new = self._hess(F, statevars, **kwargs)
        else:
            d2WdFdF = self._hess(F, **kwargs)
        return [np.array(d2WdFdF)]

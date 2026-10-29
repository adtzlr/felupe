# SPDX-License-Identifier: GPL-3.0-or-later

import jax
import numpy as np

from .._material import Material, takes_time_increment
from ._helpers import as_total_lagrange, time_increment_as_array, vmap2


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

        def viscoelastic(C, statevars, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            Cu = jnp.linalg.det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            Cin = jnp.eye(3) + statevars.reshape(3, 3)
            Ci = Cin + mu / eta * dtime * Cu
            Ci = jnp.linalg.det(Ci) ** (-1 / 3) * Ci
            invCi = jnp.linalg.inv(Ci)

            # strain energy function
            ψ = mu / 2 * (jnp.trace(Cu @ invCi) - 3)

            # incremental dissipation potential (no evolution for a zero time increment)
            Δt = jnp.where(dtime > 0, dtime, 1.0)
            φ = eta / (2 * Δt) * (jnp.trace(Cin @ invCi) - 3)
            φ = jnp.where(dtime > 0, φ, 0.0)

            # incremental potential and new state variables
            return ψ + φ, (Ci - jnp.eye(3)).ravel()

        umat = mat.Hyperelastic(viscoelastic, mu=1, eta=1, nstatevars=9)

    The state variables are the components of the deviation of the inelastic right
    Cauchy-Green deformation tensor from the identity, i.e. the initial state variables
    are zero. The function returns the incremental potential, i.e. the sum of the strain
    energy density function :math:`\psi` and the incremental dissipation potential
    :math:`\phi`, see Eq. :eq:`jax-incremental-potential`.

    ..  math::
        :label: jax-incremental-potential

        W &= \psi + \phi

        \phi &= \frac{\eta}{2 \Delta t} \left( \text{tr}\left(
            \boldsymbol{C}_{i,n} \boldsymbol{C}_i^{-1}
        \right) - 3 \right)

    The updated state variables are the stationary point of the incremental potential
    w.r.t. the unimodular tensor :math:`\boldsymbol{C}_i`. Hence, its gradient, the
    stress, equals the partial derivative of the strain energy density function at fixed
    state variables and its hessian is the consistent linearization. Otherwise, the
    stress would also include the dependence of the updated state variables on the right
    Cauchy-Green deformation tensor, see
    :func:`~felupe.constitution.tensortrax.models.hyperelastic.finite_strain_viscoelastic`.
    A zero time increment, i.e. no evolution, is handled by :func:`jax.numpy.where`,
    because the arguments of the just-in-time compiled functions are traced, i.e. their
    values are not available in Python conditions.

    **Time-dependent materials**: The function has an argument ``dtime`` without a
    default value, which is not given as keyword-argument of the material. Hence,
    the material is time-dependent (the attribute ``time_dependent`` is True), i.e.
    the time increment is an input of the material, like the deformation gradient.
    It is given by the keyword-argument ``dtime`` of :meth:`gradient` and
    :meth:`hessian`, e.g. by the time increment of a solid body which is updated by
    a :class:`~felupe.TimeStep`.
    It is passed as an array of data type float to the just-in-time compiled
    functions, i.e. they are compiled only once for all time increments.

    ..  code-block::

        assert umat.time_dependent

        P, statevars_new = umat.gradient([F, statevars], dtime=0.1)

    A time increment which is given as keyword-argument of the material, e.g.
    ``mat.Hyperelastic(viscoelastic, mu=1, eta=1, dtime=1, nstatevars=9)``, is a
    constant material parameter.

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
        >>> def neo_hooke(C, mu):
        ...     "Strain energy function of the Neo-Hookean material formulation."
        ...     return mu / 2 * (jnp.linalg.det(C) ** (-1/3) * jnp.trace(C) - 3)
        >>>
        >>> umat = mat.Hyperelastic(neo_hooke, mu=1)
        >>> ax = umat.plot(incompressible=True)

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

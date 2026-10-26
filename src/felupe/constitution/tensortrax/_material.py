# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import tensortrax as tr

from .._base import ConstitutiveMaterial
from .._material import Material as MaterialDefault
from .._material import takes_time_increment


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
        A flag to evaluate the gradient and the hessian in chunks of cells by a pool of
        threads (default is False). May introduce additional overhead for small-sized
        problems.
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

    **Time-dependent materials**: If the function has an argument ``dtime`` without a
    default value, which is not given as keyword-argument of the material, the material
    is time-dependent (the attribute ``time_dependent`` is True). Then, the time
    increment is an input of the material, like the deformation gradient, i.e. it is
    given by the keyword-argument ``dtime`` of :meth:`gradient` and :meth:`hessian`.
    A time increment which is given as keyword-argument of the material, like
    ``dtime=1`` in the code-block above, is a constant material parameter.

    ..  code-block::

        umat = mat.Material(viscoelastic, mu=1, eta=1, nstatevars=6)
        assert umat.time_dependent

        P, statevars_new = umat.gradient([F, statevars], dtime=0.1)

    ..  note::
        See the `documentation of tensortrax <https://github.com/adtzlr/tensortrax>`_
        for further details.

    Examples
    --------
    ..  plot::

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

        # the time increment is an input, if it is an argument of the (original)
        # function which is not given as material parameter
        self.time_dependent = takes_time_increment(fun, self.kwargs)

    # evaluate the gradient and the hessian in chunks of cells by FElupe (instead of
    # the generic material): also the conversions of the stresses and the elasticity
    # tensors are evaluated in chunks and the chunks are evaluated by FElupe's threads
    gradient = ConstitutiveMaterial.gradient
    hessian = ConstitutiveMaterial.hessian

    def _gradient(self, x, **kwargs):
        return self._stress(x, **self.kwargs, **kwargs)

    def _hessian(self, x, **kwargs):
        return self._elasticity(x, **self.kwargs, **kwargs)

    def _stress(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        dWdF = tr.function(self.fun, wrt=0, ntrax=2)(F, *statevars, **kwargs)
        if self.nstatevars > 0:
            statevars_new = tr.function(self.fun_statevars, wrt=0, ntrax=2)(
                F, *statevars, **kwargs
            )
        else:
            statevars_new = None
        return [dWdF, statevars_new]

    def _elasticity(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        d2WdFdF = tr.jacobian(self.fun, wrt=0, ntrax=2)(F, *statevars, **kwargs)
        return [d2WdFdF]

# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import numpy as np

from ._base import ConstitutiveMaterial


def takes_time_increment(fun, kwargs):
    """Return True if a function of a material definition takes the time increment as
    input, i.e. if it has an argument ``dtime`` without a default value, which is not
    given in the keyword-arguments of the material parameters.

    Parameters
    ----------
    fun : callable
        The function of the material definition. The signature of a decorated function
        is taken from the original function (by ``functools.wraps``).
    kwargs : dict
        The keyword-arguments of the material parameters.

    Returns
    -------
    bool
        True if the function takes the time increment as input.
    """

    if "dtime" in kwargs:
        return False  # a time increment given as material parameter

    try:
        parameter = inspect.signature(fun).parameters.get("dtime")
    except (TypeError, ValueError):  # no signature available
        return False

    keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

    return (
        parameter is not None
        and parameter.kind in keyword
        and parameter.default is inspect.Parameter.empty
    )


class Material(ConstitutiveMaterial):
    r"""A user-defined material definition with given functions for the (first
    Piola-Kirchhoff) stress tensor :math:`\boldsymbol{P}`, optional constraints on
    additional fields (e.g. :math:`p` and :math:`J`), updated state variables
    :math:`\boldsymbol{\zeta}` as well as the according fourth-order elasticity tensor
    :math:`\mathbb{A}` and the linearizations of the constraint equations. Both
    functions take a list of the 3x3 deformation gradient :math:`\boldsymbol{F}` and
    optional vector of state variables :math:`\boldsymbol{\zeta}_n` as the first input
    argument. The stress-function must return the updated state variables
    :math:`\boldsymbol{\zeta}`.

    Parameters
    ----------
    stress : callable
        A constitutive material definition which returns a list containing the (first
        Piola-Kirchhoff) stress tensor, optional additional constraints as well as the
        state variables. The state variables must always be included even if they are
        None. See template code-blocks for the required function signature.
    elasticity : callable
        A constitutive material definition which returns a list containing the fourth-
        order elasticity tensor as the jacobian of the (first Piola-Kirchhoff) stress
        tensor w.r.t. the deformation gradient, optional linearizations of the
        additional constraints. The state variables must not be returned. See template
        code-blocks for the required function signature.
    nstatevars : int, optional
        Number of internal state variable components (default is 0). State variable
        components must always be concatenated into a 1d-array.

    Notes
    -----

    ..  note::
        The first item in the list of the input arguments always contains the
        gradient of the (displacement) field :math:`\boldsymbol{u}` w.r.t. the
        undeformed coordinates  :math:`\boldsymbol{X}`. The identity matrix
        :math:`\boldsymbol{1}` is added to this gradient, i.e. the first item of the
        list ``x`` contains the deformation gradient :math:`\boldsymbol{F} =
        \boldsymbol{1} + \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}}`. All
        other fields are provided as interpolated values (no gradients evaluated).

    For :math:`(\boldsymbol{u})` single-field formulations, the callables for
    ``stress`` and ``elasticity`` must return the gradient and hessian of the strain
    energy density function :math:`\psi(\boldsymbol{F})` w.r.t. the deformation
    gradient tensor :math:`\boldsymbol{F}`.

    ..  math::

        \text{stress}(\boldsymbol{F}, \boldsymbol{\zeta}_n) = \begin{bmatrix}
            \frac{\partial \psi}{\partial \boldsymbol{F}} \\
            \boldsymbol{\zeta}
        \end{bmatrix}

    The callable for ``elasticity`` (hessian) must not return the updated state
    variables.

    ..  math::

        \text{elasticity}(\boldsymbol{F}, \boldsymbol{\zeta}_n) = \begin{bmatrix}
            \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}
        \end{bmatrix}

    Take this code-block as template:

    ..  code-block::

        def stress(x, **kwargs):
            "First Piola-Kirchhoff stress tensor."

            # extract variables
            F, statevars = x[0], x[-1]

            # user code for (first Piola-Kirchhoff) stress tensor
            P = None

            # update state variables
            statevars_new = None

            return [P, statevars_new]

        def elasticity(x, **kwargs):
            "Fourth-order elasticity tensor."

            # extract variables
            F, statevars = x[0], x[-1]

            # user code for fourth-order elasticity tensor
            # according to the (first Piola-Kirchhoff) stress tensor
            dPdF = None

            return [dPdF]

        umat = Material(stress, elasticity, **kwargs)

    For :math:`(\boldsymbol{u}, p, J)` mixed-field formulations, the callables for
    ``stress`` and ``elasticity`` must return the gradients and hessians of the
    (augmented) strain energy density function w.r.t. the deformation gradient and the
    other fields.

    ..  math::

        \text{stress}(\boldsymbol{F}, p, J, \boldsymbol{\zeta}_n) = \begin{bmatrix}
            \frac{\partial \psi}{\partial \boldsymbol{F}} \\
            \frac{\partial \psi}{\partial p} \\
            \frac{\partial \psi}{\partial J} \\
            \boldsymbol{\zeta}
        \end{bmatrix}

    For the hessians, the upper-triangle blocks have to be provided.

    ..  math::

        \text{elasticity}(\boldsymbol{F}, p, J, \boldsymbol{\zeta}_n) = \begin{bmatrix}
            \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}} \\
            \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial p} \\
            \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial J} \\
            \frac{\partial^2 \psi}{\partial p\ \partial p} \\
            \frac{\partial^2 \psi}{\partial p\ \partial J} \\
            \frac{\partial^2 \psi}{\partial J\ \partial J}
        \end{bmatrix}

    For :math:`(\boldsymbol{u}, p, J)` mixed-field formulations, take this code-block as
    template:

    ..  code-block::

        def gradient(x, **kwargs):
            "Gradients of the strain energy density function."

            # extract variables
            F, p, J, statevars = x[0], x[1], x[2], x[-1]

            # user code
            dWdF = None  # first Piola-Kirchhoff stress tensor
            dWdp = None
            dWdJ = None

            # update state variables
            statevars_new = None

            return [dWdF, dWdp, dWdJ, statevars_new]

        def hessian(x, **kwargs):
            "Hessians of the strain energy density function."

            # extract variables
            F, p, J, statevars = x[0], x[1], x[2], x[-1]

            # user code
            d2WdFdF = None  # fourth-order elasticity tensor
            d2WdFdp = None
            d2WdFdJ = None
            d2Wdpdp = None
            d2WdpdJ = None
            d2WdJdJ = None

            # upper-triangle items of the hessian
            return [d2WdFdF, d2WdFdp, d2WdFdJ, d2Wdpdp, d2WdpdJ, d2WdJdJ]

        umat = Material(gradient, hessian, **kwargs)

    **Time-dependent materials**: If the function ``stress`` has an argument ``dtime``
    without a default value, which is not given as keyword-argument of the material,
    the material is time-dependent (the attribute ``time_dependent`` is True). Then, the
    time increment is an input of the material, like the deformation gradient, i.e. it
    is given by the keyword-argument ``dtime`` of :meth:`gradient` and :meth:`hessian`,
    e.g. ``umat.gradient(x, dtime=0.1)``, and it is passed to both functions. A time
    increment which is given as keyword-argument of the material, e.g.
    ``Material(stress, elasticity, dtime=1.0)``, is a constant material parameter.

    ..  code-block::

        def stress(x, eta, dtime):
            ...

        def elasticity(x, eta, dtime):
            ...

        umat = Material(stress, elasticity, eta=1.0)
        assert umat.time_dependent

    Examples
    --------
    The compressible isotropic hyperelastic Neo-Hookean material formulation is given
    by the strain energy density function

    .. math::

        \psi(\boldsymbol{C}) = \frac{\mu}{2} \text{tr}(\boldsymbol{C})
            - \mu \ln(J) + \frac{\lambda}{2} \ln(J)^2

    with the determinant of the deformation gradient and the right Cauchy Green
    deformation tensor.

    .. math::

       J &= \text{det}(\boldsymbol{F})

       C &= \boldsymbol{F}^T\ \boldsymbol{F}

    The first Piola-Kirchhoff stress tensor is evaluated as the gradient
    of the strain energy density function.

    .. math::

       \boldsymbol{P} &= \frac{\partial \psi}{\partial \boldsymbol{F}}

       \boldsymbol{P} &= \mu \left( \boldsymbol{F} - \boldsymbol{F}^{-T} \right)
           + \lambda \ln(J) \boldsymbol{F}^{-T}

    The hessian of the strain energy density function enables the corresponding
    elasticity tensor.

    .. math::

       \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F}\ \partial
       \boldsymbol{F}}

       \mathbb{A} &= \mu \boldsymbol{I} \overset{ik}{\otimes} \boldsymbol{I}
           + \left(\mu - \lambda \ln(J) \right)
               \boldsymbol{F}^{-T} \overset{il}{\otimes} \boldsymbol{F}^{-T}
           + \lambda \boldsymbol{F}^{-T} {\otimes} \boldsymbol{F}^{-T}

    ..  plot::
        :context: close-figs

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> from felupe.math import (
        ...     cdya_ik,
        ...     cdya_il,
        ...     det,
        ...     dya,
        ...     identity,
        ...     inv,
        ...     transpose
        ... )
        >>>
        >>> def stress(x, mu, lmbda):
        ...     F, statevars = x[0], x[-1]
        ...     J = det(F)
        ...     lnJ = np.log(J)
        ...     iFT = transpose(inv(F, J))
        ...     dWdF = mu * (F - iFT) + lmbda * lnJ * iFT
        ...     return [dWdF, statevars]
        >>>
        >>> def elasticity(x, mu, lmbda):
        ...     F = x[0]
        ...     J = det(F)
        ...     iFT = transpose(inv(F, J))
        ...     eye = identity(F)
        ...     return [
        ...         mu * cdya_ik(eye, eye) + lmbda * dya(iFT, iFT) +
        ...         (mu - lmbda * np.log(J)) * cdya_il(iFT, iFT)
        ...     ]
        >>>

    The material formulation is tested in a minimal example of non-homogeneous uniaxial
    tension.

    ..  plot::
        :context: close-figs

        >>> mesh = fem.Cube(n=3)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> umat = fem.Material(stress, elasticity, mu=1.0, lmbda=2.0)
        >>> solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
        >>>
        >>> boundaries = fem.dof.uniaxial(
        ...     field, clamped=True, move=0.5, return_loadcase=False
        ... )
        >>>
        >>> step = fem.Step(items=[solid], boundaries=boundaries)
        >>> job = fem.Job(steps=[step]).evaluate()

    See Also
    --------
    felupe.NeoHookeCompressible : Nearly-incompressible isotropic hyperelastic
        Neo-Hookean material formulation.

    """

    def __init__(self, stress, elasticity, nstatevars=0, **kwargs):
        self.umat = {"gradient": stress, "hessian": elasticity}
        self.kwargs = kwargs
        self.nstatevars = nstatevars
        self.x = [np.eye(3), np.zeros(nstatevars)]
        self.time_dependent = takes_time_increment(stress, kwargs)

    def gradient(self, x, **kwargs):
        """Return the evaluated gradient of the strain energy density function.

        Parameters
        ----------
        x : list of ndarray
            The list with input arguments. These contain the extracted fields of a
            :class:`~felupe.FieldContainer` along with the old vector of state
            variables, ``[*field.extract(), statevars_old]``.
        **kwargs : dict, optional
            Optional keyword-arguments for the function, e.g. the time increment
            ``dtime`` of a time-dependent material.

        Returns
        -------
        list of ndarray
            A list with the evaluated gradient(s) of the strain energy density function
            and the updated vector of state variables.
        """

        self._check_time_increment(kwargs)

        return self.umat["gradient"](x, **self.kwargs, **kwargs)

    def hessian(self, x, **kwargs):
        """Return the evaluated upper-triangle components of the hessian(s) of the
        strain energy density function.

        Parameters
        ----------
        x : list of ndarray
            The list with input arguments. These contain the extracted fields of a
            :class:`~felupe.FieldContainer` along with the old vector of state
            variables, ``[*field.extract(), statevars_old]``.
        **kwargs : dict, optional
            Optional keyword-arguments for the function, e.g. the time increment
            ``dtime`` of a time-dependent material.

        Returns
        -------
        list of ndarray
            A list with the evaluated upper-triangle components of the hessian(s) of the
            strain energy density function.
        """

        self._check_time_increment(kwargs)

        return self.umat["hessian"](x, **self.kwargs, **kwargs)

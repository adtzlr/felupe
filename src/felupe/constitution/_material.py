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

from ._base import ConstitutiveMaterial


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
        A constitutive material definition which returns a list containting the (first
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

    ..  pyvista-plot::
        :context:
    
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
    
    ..  pyvista-plot::
        :context:

        >>> mesh = fem.Cube(n=3)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> umat = fem.Material(stress, elasticity, mu=1.0, lmbda=2.0)
        >>> solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
        >>>
        >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, move=0.5)
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

    def gradient(self, x):
        """Return the evaluated gradient of the strain energy density function.

        Parameters
        ----------
        x : list of ndarray
            The list with input arguments. These contain the extracted fields of a
            :class:`~felupe.FieldContainer` along with the old vector of state
            variables, ``[*field.extract(), statevars_old]``.

        Returns
        -------
        list of ndarray
            A list with the evaluated gradient(s) of the strain energy density function
            and the updated vector of state variables.
        """

        return self.umat["gradient"](x, **self.kwargs)

    def hessian(self, x):
        """Return the evaluated upper-triangle components of the hessian(s) of the
        strain energy density function.

        Parameters
        ----------
        x : list of ndarray
            The list with input arguments. These contain the extracted fields of a
            :class:`~felupe.FieldContainer` along with the old vector of state
            variables, ``[*field.extract(), statevars_old]``.

        Returns
        -------
        list of ndarray
            A list with the evaluated upper-triangle components of the hessian(s) of the
            strain energy density function.
        """

        return self.umat["hessian"](x, **self.kwargs)

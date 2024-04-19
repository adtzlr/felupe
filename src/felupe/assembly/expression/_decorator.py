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

from ._expression import FormExpression


def FormExpressionDecorator(v, u=None, dx=None, kwargs=None, parallel=False):
    r"""A linear or bilinear form object as function decorator on a weak-form
    with methods for integration and assembly of vectors or sparse matrices.

    Parameters
    ----------
    v : FieldContainer
        A container for the ``v`` fields. May be updated during integration / assembly.
    u : FieldContainer
        A container for the ``u`` fields. May be updated during integration / assembly.
    dx : ndarray or None, optional
        Array with (numerical) differential volumes  (default is None).
    kwargs : dict or None, optional
        Dictionary with initial optional weakform-keyword-arguments. May be
        updated during integration / assembly (default is None).

    Returns
    -------
    FormExpression
        A form object with methods for integration and assembly.

    Notes
    -----
    **Linear Form**

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    **Bilinear Form**

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Examples
    --------
    FElupe requires a pre-evaluated array for the definition of a bilinear
    :class:`felupe.IntegralForm` object on interpolated field values or their gradients.
    While this has two benefits, namely a fast integration of the form is easy to code
    and the array may be computed in any programming language, sometimes numeric
    representations of analytic linear and bilinear form expressions may be easier in
    user-code and less error prone compared to the calculation of explicit second or
    fourth-order tensors. Therefore, FElupe provides a function decorator
    :func:`felupe.Form` as an easy-to-use high-level interface, similar to what
    `scikit-fem <https://github.com/kinnala/scikit-fem>`_ offers. The
    :func:`felupe.Form` decorator handles a field container. The form class is similar,
    but not identical in its usage compared to :class:`felupe.IntegralForm`. It requires
    a callable function (with optional arguments and keyword arguments) instead of a
    pre-computed array to be passed. The bilinear form of linear elasticity serves as a
    reference example for the demonstration on how to use this feature of FElupe. The
    stiffness matrix is assembled for a unit cube out of hexahedrons.
    
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> 
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>> 
        >>> boundaries, loadcase = fem.dof.uniaxial(field, move=0.5, clamped=True)

    The bilinear form of linear elasticity is defined as

    ..  math::

        a(v, u) = \int_\Omega 2 \mu \
            \delta\boldsymbol{\varepsilon} : \boldsymbol{\varepsilon} +
            \lambda \ \text{tr}(\delta\boldsymbol{\varepsilon}) \
                \text{tr}(\boldsymbol{\varepsilon}) \ dv

    with

    ..  math::

        \delta\boldsymbol{\varepsilon} &= \text{sym}(\text{grad}(\boldsymbol{v}))

        \boldsymbol{\varepsilon} &= \text{sym}(\text{grad}(\boldsymbol{u}))

    and implemented in FElupe closely to the analytic expression. The first two
    arguments for the callable *weak-form* function of a bilinear form are always arrays
    of field (gradients) ``(v, u)`` followed by arguments and keyword arguments.
    Optionally, the integration/assembly may be performed in parallel (threaded).
    Please note that this is only faster for relatively large systems. The weak-form
    function is decorated by :func:`felupe.Form` where the appropriate fields are linked
    to ``v`` and ``u`` along with the gradient flags for both fields. Arguments as well
    as keyword arguments of the weak-form may be defined inside the decorator or as part
    of the assembly arguments.
    
    ..  pyvista-plot::
        :context:
            
        >>> from felupe.math import ddot, trace, sym, grad
        >>> 
        >>> @fem.Form(v=field, u=field, kwargs={"μ": 1.0, "λ": 2.0})
        ... def bilinearform():
        ...     "A container for a bilinear form."
        ...
        ...     def linear_elasticity(v, u, μ, λ):
        ...         "Linear elasticity."
        ...
        ...         δε, ε = sym(grad(v)), sym(grad(u))
        ...         return 2 * μ * ddot(δε, ε) + λ * trace(δε) * trace(ε)
        ...
        ...     return [linear_elasticity,]
        >>> 
        >>> stiffness_matrix = bilinearform.assemble(v=field, u=field, parallel=False)
        >>> 
        >>> system = fem.solve.partition(
        ...     field, stiffness_matrix, dof1=loadcase["dof1"], dof0=loadcase["dof0"]
        ... )
        >>> field += fem.solve.solve(*system, ext0=loadcase["ext0"])
        >>> field.plot("Principal Values of Logarithmic Strain").show()

    """

    def form(weakform):
        return FormExpression(
            weakform=weakform,
            v=v,
            u=u,
            dx=dx,
            kwargs=kwargs,
            parallel=parallel,
        )

    return form

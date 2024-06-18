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

from ._helpers import Assemble, Results


class FormItem:
    r"""An item to be used in a :class:`felupe.Step` with bilinear and optional linear
    form objects based on weak-forms with methods for integration and assembly of
    vectors / sparse matrices.

    Parameters
    ----------
    bilinearform : Form or None, optional
        A bilinear form object (default is None). If None, the resulting matrix will be
        filled with zeros.
    linearform : Form or None, optional
        A linear form object (default is None). If None, the resulting vector will be
        filled with zeros.
    sym : bool, optional
        Flag to active symmetric integration/assembly for bilinear forms (default is
        False).
    kwargs : dict or None, optional
        Dictionary with initial optional weakform-keyword-arguments (default is None).
    ramp_item : str or int, optional
        The item of the dict of keyword arguments which will be updated in a
        :class:`~felupe.Step` (default is 0). By default, the first item of the dict
        is selected. Optionally, also the key of the item to be updated may be
        given.

    Notes
    -----
    ..  warning::
        The computational cost of weak-forms defined by :func:`~felupe.Form` is much
        higher compared to :class:`~felupe.IntegralForm`. Try to re-formulate the weak
        form and use :class:`~felupe.IntegralForm` instead if performance is relevant.

    Examples
    --------
    A :class:`~felupe.FormItem` is very flexible, e.g. it is possible to define linear-
    elastic solid bodies, boundary conditions or hyperelastic mixed-field solid bodies
    by their weak forms.


    **Linear-Elasticity**

    A :class:`~felupe.FormItem` is used to create a linear-elastic solid body.

    >>> import felupe as fem
    >>> from felupe.math import ddot, sym, trace, grad
    >>>
    >>> mesh = fem.Cube(n=11)
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
    >>>
    >>> @fem.Form(v=field, u=field)
    ... def bilinearform():
    ...     def a(v, u, μ=1.0, λ=2.0):
    ...         δε, ε = sym(grad(v)), sym(grad(u))
    ...         return 2 * μ * ddot(δε, ε) + λ * trace(δε) * trace(ε)
    ...     return [a]
    >>>
    >>> item = fem.FormItem(bilinearform, linearform=None, sym=True)
    >>> step = fem.Step(items=[item], boundaries=boundaries)
    >>> job = fem.Job(steps=[step]).evaluate()


    **Boundary Condition**

    A :class:`~felupe.FormItem` is used to create a boundary condition with externally
    applied displacements which is used as a ramped-boundary in a :class:`Step`.

    >>> import felupe as fem
    >>> import numpy as np
    >>> from felupe.math import ddot
    >>>
    >>> mesh = fem.Cube(n=4)
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries = fem.dof.symmetry(field[0])
    >>> solid = fem.SolidBody(umat=fem.NeoHookeCompressible(mu=1, lmbda=2), field=field)
    >>>
    >>> face = fem.Field(fem.RegionHexahedronBoundary(mesh, mask=mesh.x == 1), dim=3)
    >>> right = fem.FieldContainer([face])
    >>>
    >>> @fem.Form(v=right)
    ... def linearform():
    ...     def L(v, value, multiplier=100):
    ...         u = right.extract(grad=False)[0]
    ...         uext = value * np.array([1, 0, 0]).reshape(3, 1, 1)
    ...         return multiplier * ddot(v, u - uext)
    ...
    ...     return [L]
    >>>
    >>>
    >>> @fem.Form(v=right, u=right)
    ... def bilinearform():
    ...     return [lambda v, u, value, multiplier=100: multiplier * ddot(v, u)]
    >>>
    >>> move = fem.FormItem(bilinearform, linearform, kwargs={"value": 0}, ramp_item=0)
    >>> values = fem.math.linsteps([0, 1], num=5)
    >>> step = fem.Step(items=[solid, move], boundaries=boundaries, ramp={move: values})
    >>> job = fem.Job(steps=[step]).evaluate()


    **Hu-Washizu (Mixed) Three-Field Formulation**

    A :class:`~felupe.FormItem` is used to create a mixed three-field variation suitable
    for nearly-incompressible hyperelastic solid bodies.

    >>> import felupe as fem
    >>> from felupe.math import ddot, grad
    >>>
    >>> mesh = fem.Cube(n=3)
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldsMixed(region, n=3)
    >>>
    >>> @fem.Form(v=field)
    ... def linearform():
    ...     def L1(du, **kwargs):
    ...         dW = linearform.dW = kwargs["umat"].gradient(kwargs["field"].extract())
    ...         return fem.math.ddot(fem.math.grad(du), dW[0])
    ...
    ...     def L2(dp, **kwargs):
    ...         return dp[0] * linearform.dW[1]
    ...
    ...     def L3(dJ, **kwargs):
    ...         return dJ[0] * linearform.dW[2]
    ...
    ...     return [L1, L2, L3]
    >>>
    >>> @fem.Form(v=field, u=field)
    ... def bilinearform():
    ...     def a11(du, Du, **kwargs):
    ...         d2W = bilinearform.d2W = kwargs["umat"].hessian(kwargs["field"].extract())
    ...         return ddot(ddot(grad(du), d2W[0], mode=(2, 4)), grad(Du))
    ...
    ...     def a12(du, Dp, **kwargs):
    ...         return ddot(grad(du), bilinearform.d2W[1]) * Dp[0]
    ...
    ...     def a13(du, DJ, **kwargs):
    ...         return ddot(grad(du), bilinearform.d2W[2]) * DJ[0]
    ...
    ...     def a22(dp, Dp, **kwargs):
    ...         return dp[0] * bilinearform.d2W[3] * Dp[0]
    ...
    ...     def a23(dp, DJ, **kwargs):
    ...         return dp[0] * bilinearform.d2W[4] * DJ[0]
    ...
    ...     def a33(dJ, DJ, **kwargs):
    ...         return dJ[0] * bilinearform.d2W[5] * DJ[0]
    ...
    ...     return [a11, a12, a13, a22, a23, a33]
    >>>
    >>> umat = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    >>> item = fem.FormItem(bilinearform, linearform, kwargs={"umat": umat, "field": field})
    >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, move=-0.3)
    >>> step = fem.Step(items=[item], boundaries=boundaries)
    >>> job = fem.Job(steps=[step], boundary=boundaries["move"]).evaluate()

    See Also
    --------
    felupe.Form : A function decorator for a linear- or bilinear-form object.
    felupe.Step : A Step with multiple substeps.
    """

    def __init__(
        self, bilinearform=None, linearform=None, sym=False, kwargs=None, ramp_item=0
    ):
        self.bilinearform = bilinearform
        self.linearform = linearform

        self.sym = sym
        self.kwargs = kwargs
        self.ramp_item = ramp_item

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        if self.bilinearform is not None:
            self.field = self.bilinearform.form.v.field
        else:
            if self.linearform is not None:
                self.field = self.linearform.form.v.field

    def update(self, value):
        key = self.ramp_item
        if isinstance(key, int):
            key = list(self.kwargs.keys())[key]

        self.kwargs[key] = value

    def _vector(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        if self.linearform is not None:
            self.results.force = self.linearform.assemble(
                v=self.field, parallel=parallel, kwargs=self.kwargs
            )

        else:
            from scipy.sparse import csr_matrix

            self.results.force = csr_matrix((sum(self.field.fieldsizes), 1))

        return self.results.force

    def _matrix(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        if self.bilinearform is not None:
            self.results.stiffness = self.bilinearform.assemble(
                v=self.field,
                u=self.field,
                parallel=parallel,
                sym=self.sym,
                kwargs=self.kwargs,
            )

        else:
            from scipy.sparse import csr_matrix

            size = sum(self.field.fieldsizes)
            self.results.stiffness = csr_matrix((size, size))

        return self.results.stiffness

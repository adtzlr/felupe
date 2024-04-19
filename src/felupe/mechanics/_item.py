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
    args : tuple or None, optional
        Tuple with initial optional weakform-arguments (default is None).
    kwargs : dict or None, optional
        Dictionary with initial optional weakform-keyword-arguments (default is None).

    Examples
    --------
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

    See Also
    --------
    felupe.Form : A function decorator for a linear- or bilinear-form object.
    felupe.Step : A Step with multiple substeps.

    """

    def __init__(self, bilinearform=None, linearform=None, sym=False, kwargs=None):
        self.bilinearform = bilinearform
        self.linearform = linearform

        self.sym = sym
        self.kwargs = kwargs

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        if self.bilinearform is not None:
            self.field = self.bilinearform.form.v.field
        else:
            if self.linearform is not None:
                self.field = self.linearform.form.v.field

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

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
    bilinearform : Form
        A bilinear form object.
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
    >>> from felupe.math import ddot, sym, trace

    >>> mesh = fem.Cube(n=11)
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

    >>> @fem.Form(v=field, u=field, grad_v=[True], grad_u=[True])
    >>> def bilinearform():
    >>>     def a(gradv, gradu, μ=1.0, λ=2.0):
    >>>         δε, ε = sym(gradv), sym(gradu)
    >>>         return 2 * μ * ddot(δε, ε) + λ * trace(δε) * trace(ε)
    >>>     return [a]

    >>> item = fem.FormItem(bilinearform, linearform=None, sym=True)
    >>> step = fem.Step(items=[item], boundaries=boundaries)
    >>> fem.Job(steps=[step]).evaluate()

    See Also
    --------
    felupe.Form : A function decorator for a linear- or bilinear-form object.
    felupe.Step : A Step with multiple substeps.

    """

    def __init__(
        self, bilinearform, linearform=None, sym=False, args=None, kwargs=None
    ):
        self.bilinearform = bilinearform
        self.linearform = linearform

        self.sym = sym
        self.args = args
        self.kwargs = kwargs

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self.field = self.bilinearform.form.v.field

    def _vector(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        if self.linearform is not None:
            self.results.force = self.linearform.assemble(
                v=self.field, parallel=parallel, args=self.args, kwargs=self.kwargs
            )

        else:
            from scipy.sparse import csr_matrix

            self.results.force = csr_matrix(self.field.fields[0].values.shape)

        return self.results.force

    def _matrix(self, field=None, parallel=False):
        if field is not None:
            self.field = field

        self.results.stiffness = self.bilinearform.assemble(
            v=self.field,
            u=self.field,
            parallel=parallel,
            sym=self.sym,
            args=self.args,
            kwargs=self.kwargs,
        )

        return self.results.stiffness

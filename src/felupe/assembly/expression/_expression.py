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

from ._basis import Basis
from ._mixed import BilinearFormExpression, LinearFormExpression


class FormExpression:
    r"""A linear or bilinear form object based on a weak-form with methods for
    integration and assembly of vectors / sparse matrices.

    Linear Form:

    ..  math::

        L(v) = \int_\Omega f \cdot v \ dx

    Bilinear Form:

    ..  math::

        a(v, u) = \int_\Omega v \cdot f \cdot u \ dx

    Parameters
    ----------
    v : Field or FieldMixed
        An object with interpolation and gradients of a field. May be
        updated during integration / assembly.
    u : Field or FieldMixed
        An object with interpolation and gradients of a field. May be
        updated during integration / assembly.
    dx : ndarray or None, optional
        Array with (numerical) differential volumes (default is None).
    args : tuple or None, optional
        Tuple with initial optional weakform-arguments. May be updated during
        integration / assembly (default is None).
    kwargs : dict or None, optional
        Dictionary with initial optional weakform-keyword-arguments. May be updated
        during integration / assembly (default is None).
    parallel : bool, optional
        Flag to activate parallel (threaded) basis evaluation (default is False).

    """

    def __init__(
        self,
        weakform,
        v,
        u=None,
        dx=None,
        kwargs=None,
        parallel=False,
    ):
        # set attributes
        self.form = None
        self.dx = dx
        self.weakform = weakform
        self.kwargs = kwargs

        # init underlying linear or bilinear (mixed) form
        self._init_or_update_forms(v, u, kwargs, parallel)

    def _init_or_update_forms(self, v, u, kwargs, parallel):
        "Init or update the underlying form object."

        # update kwargs for weakform
        if kwargs is not None:
            self.kwargs = kwargs

        # get current form type
        if self.form is not None:
            form_type = type(self.form)
        else:
            form_type = None

        if v is not None:
            # linear form
            if u is None:
                self.u = None

                # mixed-field input
                self.v = Basis(v, parallel=parallel)
                form = LinearFormExpression(self.v)

                # evaluate weakform to list of weakforms
                if isinstance(self.weakform, type(lambda x: x)):
                    self.weakform = self.weakform()

            else:
                self.v = Basis(v, parallel=parallel)
                self.u = Basis(u, parallel=parallel)
                form = BilinearFormExpression(self.v, self.u)

                # evaluate weakform to list of weakforms
                if isinstance(self.weakform, type(lambda x: x)):
                    self.weakform = self.weakform()

            # check if new form type matches initial form type (update-stage)
            if form_type is not None:
                if form_type == type(form):
                    self.form = form
                else:
                    raise TypeError("Wrong type of fields.")

            # set form without a check (init-stage)
            else:
                self.form = form

    def integrate(self, v=None, u=None, kwargs=None, parallel=False, sym=False):
        r"""Return evaluated (but not assembled) integrals.

        Parameters
        ----------
        v : Field, FieldMixed or None, optional
            An object with interpolation and gradients of a field (default is None).
        u : Field, FieldMixed or None, optional
            An object with interpolation and gradients of a field (default is None).
        kwargs : dict, optional (default is {})
            Dictionary with optional weakform-keyword-arguments.
        parallel : bool, optional (default is False)
            Flag to activate parallel threading.
        sym : bool, optional (default is False)
            Flag to active symmetric integration/assembly
            for bilinear forms.

        Returns
        -------
        values : ndarray
            Integrated (but not assembled) vector / matrix values.
        """

        self._init_or_update_forms(v, u, kwargs, parallel)

        kwargs = dict(parallel=parallel, sym=sym)

        if self.u is None:
            kwargs.pop("sym")

        return self.form.integrate(self.weakform, kwargs=self.kwargs, **kwargs)

    def assemble(self, v=None, u=None, kwargs=None, parallel=False, sym=False):
        r"""Return the assembled integral as vector / sparse matrix.

        Parameters
        ----------
        v : Field, FieldMixed or None, optional
            An object with interpolation and gradients of a field (default is None).
        u : Field, FieldMixed or None, optional
            An object with interpolation and gradients of a field (default is None).
        kwargs : dict, optional (default is {})
            Dictionary with optional weakform-keyword-arguments.
        parallel : bool, optional (default is False)
            Flag to activate parallel threading.
        sym : bool, optional (default is False)
            Flag to active symmetric integration/assembly
            for bilinear forms.

        Returns
        -------
        values : csr_matrix
            The assembled vector / sparse matrix.
        """

        self._init_or_update_forms(v, u, kwargs, parallel)

        kwargs = dict(parallel=parallel, sym=sym)

        if self.u is None:
            kwargs.pop("sym")

        return self.form.assemble(self.weakform, kwargs=self.kwargs, **kwargs)

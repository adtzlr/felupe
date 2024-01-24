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

from ..assembly import IntegralForm
from ..constitution import AreaChange
from ..math import det, dot, transpose
from ..tools._plot import ViewSolid
from ._helpers import Assemble, Evaluate, Results


class Solid:
    "Base class for solid bodies which provides methods for visualisations."

    def view(self, point_data=None, cell_data=None, cell_type=None):
        """View the solid with optional given dicts of point- and cell-data items.

        Parameters
        ----------
        point_data : dict or None, optional
            Additional point-data dict (default is None).
        cell_data : dict or None, optional
            Additional cell-data dict (default is None).
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).

        Returns
        -------
        felupe.ViewSolid
            An object which provides visualization methods for solid bodies,
            e.g. :class:`felupe.SolidBody`.

        See Also
        --------
        felupe.ViewSolid : Visualization methods for :class:`felupe.SolidBody`.
        """

        return ViewSolid(
            self.field,
            solid=self,
            point_data=point_data,
            cell_data=cell_data,
            cell_type=cell_type,
        )

    def plot(self, *args, **kwargs):
        """Plot the solid body.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        """
        return self.view().plot(*args, **kwargs)

    def screenshot(
        self,
        *args,
        filename="solidbody.png",
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the solid body.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

    def imshow(self, *args, **kwargs):
        """Take a screenshot of the solid body, show the image data in a figure and
        return the ax.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.imshow(self.screenshot(*args, filename=None, **kwargs))
        ax.set_axis_off()

        return ax


class SolidBody(Solid):
    "A SolidBody with methods for the assembly of sparse vectors/matrices."

    def __init__(self, umat, field, statevars=None):
        self.umat = umat
        self.field = field

        self.results = Results(stress=True, elasticity=True)
        self.results.kinematics = self._extract(self.field)

        if statevars is not None:
            self.results.statevars = statevars
        else:
            self.results.statevars = np.zeros(
                (
                    *umat.x[-1].shape,
                    field.region.quadrature.npoints,
                    field.region.mesh.ncells,
                )
            )

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
            kirchhoff_stress=self._kirchhoff_stress,
        )

        self._area_change = AreaChange()

        self._form = IntegralForm

    def _vector(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        if field is not None:
            self.field = field

        self.results.stress = self._gradient(field, args=args, kwargs=kwargs)
        self.results.force = self._form(
            fun=self.results.stress[slice(items)],
            v=self.field,
            dV=self.field.region.dV,
        ).assemble(parallel=parallel)

        return self.results.force

    def _matrix(self, field=None, parallel=False, items=None, args=(), kwargs={}):
        if field is not None:
            self.field = field

        self.results.elasticity = self._hessian(field, args=args, kwargs=kwargs)

        form = self._form(
            fun=self.results.elasticity[slice(items)],
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        self.results.stiffness_values = form.integrate(
            parallel=parallel, out=self.results.stiffness_values
        )

        self.results.stiffness = form.assemble(values=self.results.stiffness_values)

        return self.results.stiffness

    def _extract(self, field):
        self.field = field
        self.results.kinematics = self.field.extract(out=self.results.kinematics)

        return self.results.kinematics

    def _gradient(self, field=None, args=(), kwargs={}):
        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(self.field)

        gradient = self.umat.gradient(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )

        self.results.stress, self.results._statevars = gradient[:-1], gradient[-1]

        return self.results.stress

    def _hessian(self, field=None, args=(), kwargs={}):
        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(self.field)

        self.results.elasticity = self.umat.hessian(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )

        return self.results.elasticity

    def _kirchhoff_stress(self, field=None):
        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]

        return dot(P, transpose(F))

    def _cauchy_stress(self, field=None):
        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]
        J = det(F)

        return dot(P, transpose(F)) / J

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

from ..dof import apply, partition
from ..tools import newtonrhapson


class Step:
    """A Step with multiple substeps, subsequently depending on the solution
    of the previous substep.

    Parameters
    ----------
    items : list of SolidBody, SolidBodyNearlyIncompressible, SolidBodyPressure, SolidBodyGravity, PointLoad, MultiPointConstraint or MultiPointContact
        A list of items with methods for the assembly of sparse vectors/matrices.
    ramp : dict, optional
        A dict with :class:`~felupe.Boundary` or ``item``-keys which holds the array of
        values to ramp (default is None). If None, only one substep is evaluated.
    boundaries : dict of Boundary, optional
        A dict with :class:`~felupe.Boundary` conditions (default is None).

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> boundaries = fem.dof.symmetry(field[0])
        >>> boundaries["clamped"] = fem.Boundary(field[0], fx=1, skip=(True, False, False))
        >>> boundaries["move"] = fem.Boundary(field[0], fx=1, skip=(False, True, True))
        >>>
        >>> umat = fem.NeoHooke(mu=1, bulk=2)
        >>> solid = fem.SolidBody(umat, field)
        >>>
        >>> move = fem.math.linsteps([0, 1], num=5)
        >>> step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
        >>>
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> ax = solid.plot("Principal Values of Cauchy Stress").show()
    """

    def __init__(self, items, ramp=None, boundaries=None):
        self.items = items

        if ramp is None:
            self.ramp = {}
            self.nsubsteps = 1
        else:
            self.ramp = dict(ramp)
            self.nsubsteps = len(list(self.ramp.values())[0])

        if boundaries is None:
            boundaries = {}

        self.boundaries = boundaries

    def generate(self, **kwargs):
        "Yield all generated substeps."

        substeps = np.arange(self.nsubsteps)

        if "x0" not in kwargs.keys():
            field = self.items[0].field
        else:
            field = kwargs["x0"]

        stop = False
        for substep in substeps:
            if stop:
                break

            # update items
            for item, value in self.ramp.items():
                item.update(value[substep])

            # update load case
            dof0, dof1 = partition(field, self.boundaries)
            ext0 = apply(field, self.boundaries, dof0)

            # run newton-rhapson iterations
            res = newtonrhapson(
                items=self.items,
                dof0=dof0,
                dof1=dof1,
                ext0=ext0,
                **kwargs,
            )

            if not res.success:
                stop = True
                break
            else:
                yield res

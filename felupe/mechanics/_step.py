# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from ..dof import partition, apply
from ..tools import newtonrhapson


class Step:
    """A Step with multiple substeps, subsequently depending on the solution
    of the previous substep."""

    def __init__(self, items, ramp, boundaries):
        """A Step with multiple substeps, subsequently depending on the solution
        of the previous substep."""

        self.items = items
        self.ramp = dict(ramp)
        self.boundaries = boundaries
        self.nsubsteps = len(list(self.ramp.values())[0])

    def generate(self, **kwargs):
        "Generate all substeps."

        substeps = np.arange(self.nsubsteps)

        if not "x0" in kwargs.keys():
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

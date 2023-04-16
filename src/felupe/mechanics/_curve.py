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

from ..tools import force
from ._job import Job


class CharacteristicCurve(Job):
    def __init__(
        self,
        steps,
        boundary,
        items=None,
        callback=lambda stepnumber, substepnumber, substep: None,
    ):

        super().__init__(steps, self._callback)

        self.items = items
        self.boundary = boundary
        self.x = []
        self.y = []
        self.res = None
        self._cb = callback

    def _callback(self, stepnumber, substepnumber, substep):

        if self.items is not None:
            fun = sum([item.results.force for item in self.items])
        else:
            fun = substep.fun

        self.x.append(substep.x[0].values[self.boundary.points[0]])
        self.y.append(force(substep.x, fun, self.boundary))
        self.res = substep

        self._cb(stepnumber, substepnumber, substep)

    def plot(
        self,
        x=None,
        y=None,
        xaxis=0,
        yaxis=0,
        xlabel="x",
        ylabel="y",
        xscale=1,
        yscale=1,
        gradient=False,
        fig=None,
        ax=None,
        items=None,
        **kwargs
    ):
        if self.res is None:
            raise ValueError(
                "Results are empty. Run `job.evaluate()` and call `job.plot()` again."
            )

        import matplotlib.pyplot as plt

        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if items is None:
            items = slice(None)

        x = np.array(x)[items]
        y = np.array(y)[items]

        if gradient:
            y = np.gradient(y, x[:, xaxis], edge_order=2, axis=0)
            z = np.gradient(y, x[:, xaxis], edge_order=2, axis=0)
            cuttoff = np.mean(abs(z[:, yaxis])) + 2 * np.std(abs(z[:, yaxis]))
            y[abs(z) > cuttoff] = np.nan

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(x[:, xaxis] * xscale, y[:, yaxis] * yscale, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

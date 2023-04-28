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


def linsteps(points, num=10, endpoint=True, axis=None, axes=None):

    points = np.array(points).ravel()
    start = points[:-1]
    end = points[1:]
    num = np.array([num]).ravel()

    if len(num) == 1:
        num = np.tile(num, max(1, len(start)))

    num = np.pad(num, (0, max(0, len(start) - len(num))), mode="edge")

    steplist = [
        np.linspace(a, b, n, endpoint=False) for a, b, n in zip(start, end, num)
    ]
    if len(steplist) > 0:
        steps = np.concatenate(
            [np.linspace(a, b, n, endpoint=False) for a, b, n in zip(start, end, num)]
        )
    else:
        steps = np.array([])

    if endpoint:
        steps = np.append(steps, points[-1])

    if axis is not None:

        if axes is None:
            axes = axis + 1

        steps_1d = steps
        steps = np.zeros((len(steps_1d), axes))
        steps[:, axis] = steps_1d

    return steps

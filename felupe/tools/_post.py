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
from scipy.interpolate import interp1d


def force(field, r, boundary, unstack=None):
    if unstack is None:
        return ((r.reshape(-1, 3))[boundary.points]).sum(0)
    else:
        return (((np.split(r, unstack)[0]).reshape(-1, 3))[boundary.points]).sum(0)


def moment(field, r, boundary, point=np.zeros(3), unstack=None):

    point = point.reshape(1, 3)

    indices = np.array([(1, 2), (2, 0), (0, 1)])

    displacements = field.values
    d = ((point + displacements) - point)[boundary.points]

    if unstack is None:
        force = r.reshape(-1, 3)
    else:
        force = (np.split(r, unstack)[0]).reshape(-1, 3)

    f = force[boundary.points]

    return np.array([(f[:, i] * d[:, i[::-1]]).sum() for i in indices])


def curve(x, y):
    kind = [None, "linear", "quadratic", "cubic"][min(len(y), 4) - 1]
    f = interp1d(x[: len(y)], y, kind=kind)
    xx = np.linspace(x[0], x[: len(y)][-1])
    return np.array([x[: len(y)], y]), np.array([xx, f(xx)])

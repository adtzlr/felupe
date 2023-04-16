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

from ._tools import expand


def line_line(a=0, b=1, n=2):
    "Line generator."
    points = np.linspace(a, b, n).reshape(-1, 1)
    cells = np.repeat(np.arange(n), 2)[1:-1].reshape(-1, 2)
    cell_type = "line"

    return points, cells, cell_type


def rectangle_quad(a=(0, 0), b=(1, 1), n=(2, 2)):
    "Rectangle generator."
    dim = 2
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of points per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    line = line_line(a=a[0], b=b[0], n=n[0])

    points, cells, cell_type = expand(*line, n=n[-1], z=b[-1] - a[-1])
    points[:, -1] += a[-1]

    return points, cells, cell_type


def cube_hexa(a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
    "Cube generator."
    dim = 3
    array_like = (tuple, list, np.ndarray)

    # check if number "n" is scalar or no. of points per axis (array-like)
    if not isinstance(n, array_like):
        n = np.full(dim, n, dtype=int)

    rectangle = rectangle_quad(a=a[:-1], b=b[:-1], n=n[:-1])

    points, cells, cell_type = expand(*rectangle, n=n[-1], z=b[-1] - a[-1])
    points[:, -1] += a[-1]

    return points, cells, cell_type

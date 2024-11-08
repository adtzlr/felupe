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
from functools import wraps

from ....tensortrax.models.hyperelastic import miehe_goektepe_lulei as mgl_docstring
from .microsphere import langevin, linear, nonaffine_stretch, nonaffine_tube


@wraps(mgl_docstring)
def miehe_goektepe_lulei(C, mu, N, U, p, q):
    kwargs_stretch = {"mu": mu, "N": N}
    kwargs_tube = {"mu": mu * N * U}

    return nonaffine_stretch(
        C, p=p, f=langevin, kwargs=kwargs_stretch
    ) + nonaffine_tube(C, q=q, f=linear, kwargs=kwargs_tube)

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

from ....tensortrax.models.lagrange import morph_representative_directions as morph_repr
from ._morph_uniaxial import morph_uniaxial
from .microsphere import affine_force_statevars


@wraps(morph_repr)
def morph_representative_directions(F, statevars, p, ε=1e-6):
    def f(λ, statevars, **kwargs):
        dψdλ, statevars_new = morph_uniaxial(λ, statevars, **kwargs)
        return 5 * dψdλ, statevars_new

    return affine_force_statevars(F, statevars, f=f, kwargs={"p": p, "ε": ε})

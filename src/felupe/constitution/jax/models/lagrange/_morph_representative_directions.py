# SPDX-License-Identifier: GPL-3.0-or-later

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

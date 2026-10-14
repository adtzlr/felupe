# SPDX-License-Identifier: GPL-3.0-or-later

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

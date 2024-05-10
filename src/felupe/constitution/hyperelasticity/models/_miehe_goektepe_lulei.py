from tensortrax.math.linalg import det

from ....quadrature import BazantOh
from .microsphere import langevin, linear, nonaffine_stretch, nonaffine_tube


def miehe_goektepe_lulei(C, mu, N, U, p, q):
    """Micro-sphere model: Combined non-affine stretch and
    tube model (for details see Miehe, Goektepe and Lulei (2004))."""

    kwargs_stretch = {"mu": mu, "N": N}
    kwargs_tube = {"mu": mu * N * U}

    scheme = BazantOh(n=21)

    return nonaffine_stretch(
        C, p=p, f=langevin, kwargs=kwargs_stretch, quadrature=scheme
    ) + nonaffine_tube(C, q=q, f=linear, kwargs=kwargs_tube, quadrature=scheme)

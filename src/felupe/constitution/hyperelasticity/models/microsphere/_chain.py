from ...math import log, sinh, sqrt


def langevin(stretch, mu, N):
    """Langevin model given by the free energy of a single chain as a function
    of the stretch (assuming a complex valued logarithm). The inverse
    Langevin function is defined by a Padé approximation."""

    x = stretch / sqrt(N)
    L = x * (3 - x**2) / (1 - x**2)

    return mu * N * (x * L + log(L / sinh(L)))


def langevin2(stretch, mu, N):
    """Langevin model given by the free energy of a single chain as a function
    of the stretch (assuming a complex valued logarithm). The inverse Langevin
    function is defined by a Padé approximation.

    **Note**: This function is optimized for fast gradient evaluation but
    cannot be used to calculate the actual free energy - only its derivative
    w.r.t. the stretch gives a real-valued result for the region of interest
    `N > stretch ** 2`."""

    return mu * (stretch**2 / 2 - N * log(stretch**2 - N))


def gauss(stretch, mu):
    """Gaussian model given by the free energy
    of a single chain as a function of the stretch."""

    return 3 * mu / 2 * (stretch**2 - 1)


def linear(stretch, mu):
    """Linear model given by the free energy
    of a single chain as a function of the stretch."""

    return mu * (stretch - 1)

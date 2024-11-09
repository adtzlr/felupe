from jax.numpy import log, sinh, sqrt


def langevin(stretch, mu, N):
    """Langevin model given by the free energy of a single chain as a function of the
    stretch (assuming a complex valued logarithm). The inverse Langevin function is
    defined by a Padé approximation.
    """

    x = stretch / sqrt(N)
    L = x * (3 - x**2) / (1 - x**2)

    return mu * N * (x * L + log(L / sinh(L)))


def linear(stretch, mu):
    """Linear model given by the free energy
    of a single chain as a function of the stretch."""

    return mu * (stretch - 1)

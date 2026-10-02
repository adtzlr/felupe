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
import inspect
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import det

# Perturbation to separate repeated eigenvalues. A fixed perturbation can't separate
# a repeated eigenvalue if its eigenspace is a plane on which the perturbation is
# isotropic. For diag(1, -1, 0), these are the planes with normals (1, +/-1, 0), e.g.
# uniaxial loading at 45° in the xy-plane. This generic matrix with eigenvalues
# (-1, 0, 1) moves these planes to directions without any symmetry. This is the same
# perturbation as used in tensortrax.
PERTURBATION = np.array(
    [
        [0.03, 0.44, -0.66],
        [0.44, 0.55, -0.24],
        [-0.66, -0.24, -0.58],
    ]
)


def perturb(A, eps=None):
    r"""Add a relative perturbation, scaled by the norm of a symmetric matrix ``A``, to
    separate repeated eigenvalues.

    Parameters
    ----------
    A : jax.Array
        A symmetric matrix or a batch of symmetric matrices of shape ``(..., M, M)``
        with ``M <= 3``.
    eps : float or None, optional
        The relative magnitude of the perturbation. Default is None, where the square
        root of the machine epsilon is used in double precision (as in tensortrax) and
        ``1e-4`` otherwise.

    Returns
    -------
    jax.Array
        The perturbed matrix (or batch of matrices).

    Notes
    -----
    The perturbation is scaled by the Frobenius norm of the matrix (or by one, if the
    norm is zero) and it is treated as a constant, i.e. it does not contribute to the
    derivatives. The derivatives are those of the unperturbed matrix, evaluated at the
    perturbed matrix.

    The perturbation shifts the eigenvalues (bias, proportional to ``eps``) and the
    round-off of the separated eigenvalues enters the second derivatives (proportional
    to the machine epsilon divided by ``eps``). In single precision, the square root of
    the machine epsilon (approx. ``3.5e-4``) leads to an unnecessarily large bias,
    whereas ``1e-4`` balances both errors better.
    """

    x = jax.lax.stop_gradient(A)

    if eps is None:
        if jnp.finfo(x.dtype).bits >= 64:
            eps = np.sqrt(jnp.finfo(x.dtype).eps)
        else:
            eps = 1e-4

    dim = x.shape[-1]
    norm = jnp.sqrt(jnp.sum(x**2, axis=(-2, -1), keepdims=True))
    scale = eps * jnp.where(norm > 0, norm, 1.0)

    return A + scale * jnp.asarray(PERTURBATION[:dim, :dim], dtype=x.dtype)


def eigvalsh(A, eps=None):
    r"""Return the eigenvalues (in ascending order) of a symmetric matrix ``A``, which
    is perturbed (relative to its norm) to separate repeated eigenvalues.

    Parameters
    ----------
    A : jax.Array
        A symmetric matrix or a batch of symmetric matrices of shape ``(..., M, M)``
        with ``M <= 3``.
    eps : float or None, optional
        The relative magnitude of the perturbation. Default is None, see
        :func:`perturb`.

    Returns
    -------
    jax.Array
        The eigenvalues in ascending order.

    See Also
    --------
    jax.numpy.linalg.eigvalsh : Compute the eigenvalues of a Hermitian matrix.
    """
    return jnp.linalg.eigvalsh(perturb(A, eps=eps))


def eigh(A, eps=None):
    r"""Return the eigenvalues (in ascending order) and eigenvectors (as columns) of a
    symmetric matrix ``A``, which is perturbed (relative to its norm) to separate
    repeated eigenvalues.

    Parameters
    ----------
    A : jax.Array
        A symmetric matrix or a batch of symmetric matrices of shape ``(..., M, M)``
        with ``M <= 3``.
    eps : float or None, optional
        The relative magnitude of the perturbation. Default is None, see
        :func:`perturb`.

    Returns
    -------
    eigenvalues : jax.Array
        The eigenvalues in ascending order.
    eigenvectors : jax.Array
        The normalized eigenvectors, where the column ``eigenvectors[:, i]`` is the
        eigenvector of the eigenvalue ``eigenvalues[i]``.

    See Also
    --------
    jax.numpy.linalg.eigh : Compute the eigenvalues and eigenvectors of a Hermitian
        matrix.
    """
    return jnp.linalg.eigh(perturb(A, eps=eps))


def vmap(fun, in_axes=0, out_axes=0, method=jax.vmap, **kwargs):
    """Vectorizing map. Creates a function which maps ``fun`` over argument axes. This
    decorator treats all non-specified arguments and keyword-arguments as static.

    See Also
    --------
    jax.vmap : Vectorizing map. Creates a function which maps ``fun`` over argument
        axes.
    """

    @wraps(fun)
    def vmap_with_static_kwargs(*args, **keywordargs):
        # sorted list of all parameter keys, including kwargs with default values
        sig = inspect.signature(fun)
        keys = [
            key
            for key, value in sig.parameters.items()
            if not (key in ["args", "kwargs"] and value.default == inspect._empty)
        ]

        if not (
            "kwargs" in sig.parameters.keys()
            and sig.parameters["kwargs"].default == inspect._empty
        ):
            # check if unexpected keyword-argument is given
            for key in keywordargs.keys():
                if key not in keys:
                    raise TypeError(
                        f"{fun.__name__}() got an unexpected keyword argument '{key}'"
                    )

        # dict with default values for all parameters
        parameters = dict(
            [(key, value.default) for key, value in sig.parameters.items()]
        )

        # merge dict of default values with custom keyword arguments
        items = {**parameters, **keywordargs}

        # create sorted list of values of keyword-arguments, including default kwargs
        keyword_args = [items[key] for key in keys[len(args) :]]

        # don't map non-given arguments and keyword-arguments
        if not hasattr(in_axes, "__len__"):
            in_axes_tuple = (in_axes,)
        else:
            in_axes_tuple = in_axes

        static_argnums = len(args) + len(keyword_args) - len(in_axes_tuple)
        in_axes_new = (*in_axes_tuple, *([None] * static_argnums))

        vfun = method(fun, in_axes=in_axes_new, out_axes=out_axes, **kwargs)

        return vfun(*args, *keyword_args)

    return vmap_with_static_kwargs


def vmap2(fun, in_axes=[0, 0], out_axes=[0, 0], methods=[jax.vmap, jax.vmap], **kwargs):
    "Nested vectorizing map."
    return vmap(
        vmap(
            fun, in_axes=in_axes[0], out_axes=out_axes[0], method=methods[0], **kwargs
        ),
        in_axes=in_axes[1],
        out_axes=out_axes[1],
        method=methods[1],
        **kwargs,
    )


def as_total_lagrange(fun):
    @wraps(fun)
    def evaluate(F, *args, **kwargs):
        i, j = jnp.triu_indices(3)
        C_triu = jnp.einsum("ia,ia->a", F[:, i], F[:, j])
        C = C_triu[jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])]
        return fun(C, *args, **kwargs)

    return evaluate


def isochoric_volumetric_split(fun):
    """Apply the material formulation only on the isochoric part of the
    multiplicative split of the deformation gradient."""

    @wraps(fun)
    def apply_iso(C, *args, **kwargs):
        return fun(det(C) ** (-1 / 3) * C, *args, **kwargs)

    return apply_iso

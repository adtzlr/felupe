# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


def in_plane(out, axes):
    """Return the view of the in-plane components of a given output array of a field
    in three-dimensional space and set its out-of-plane components to zero.

    Parameters
    ----------
    out : ndarray or None
        The output array with three components on each of the first ``axes`` axes, e.g.
        of shape (3, 3, q, c) for a gradient (``axes=2``). If None, None is returned.
    axes : int
        The number of leading axes of the components.

    Returns
    -------
    ndarray or None
        The view of the in-plane components, i.e. the first two items of each of the
        first ``axes`` axes, e.g. of shape (2, 2, q, c).
    """

    if out is None:
        return None

    for axis in range(axes):
        out[(slice(None),) * axis + (slice(2, None),)] = 0

    return out[(slice(2),) * axes]


def embed(values, out, axes):
    """Embed the in-plane components of a field in three-dimensional space.

    Parameters
    ----------
    values : ndarray
        The in-plane components with two components on each of the first ``axes``
        axes. If ``out`` is given, this is the view of :func:`in_plane`.
    out : ndarray or None
        The output array, given to :func:`in_plane`. If None, the out-of-plane
        components of ``values`` are padded with zeros (a new array is returned).
    axes : int
        The number of leading axes of the components.

    Returns
    -------
    ndarray
        The array with three components on each of the first ``axes`` axes.
    """

    if out is not None:
        return out

    if axes == 0:
        return values

    return np.pad(values, [(0, 1)] * axes + [(0, 0)] * (values.ndim - axes))

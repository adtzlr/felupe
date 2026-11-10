# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import issparse

from ..math import cross


def force(field, forces, boundary):
    r"""Return the sum of the force vectors at the points of a boundary.

    Parameters
    ----------
    field : felupe.FieldContainer
        The field container. The function uses only the first field of the container.
    forces : ndarray or scipy.sparse matrix
        The assembled vector of the internal forces. For example, use ``res.fun`` of
        :func:`~felupe.newtonraphson` or ``context.substep.fun`` in a
        :class:`~felupe.Plugin`.
    boundary : felupe.Boundary
        The boundary. The point ids ``boundary.points`` select the force vectors.

    Returns
    -------
    ndarray of shape (dim,)
        The sum of the force vectors at the points of the boundary.

    See Also
    --------
    felupe.tools.moment : Return the sum of the moment vectors at the points of a
        boundary.

    Examples
    --------
    Apply uniaxial tension to a cube. Then, calculate the reaction force on the
    boundary ``"move"``.

    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> region = fem.RegionHexahedron(fem.Cube(n=3))
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, return_loadcase=True)
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)
    >>> res = fem.newtonraphson(items=[solid], **loadcase, verbose=0)
    >>>
    >>> F = fem.tools.force(field, res.fun, boundaries["move"])
    >>> print(F[0].round(3))
    0.428

    The force is parallel to the x-axis. The other components are zero.

    >>> np.allclose(F[1:], 0)
    True

    """

    if issparse(forces):
        forces = forces.toarray()

    forces_first_field = np.split(forces, field.offsets)[0]
    dim = field[0].dim

    return ((forces_first_field.reshape(-1, dim))[boundary.points]).sum(axis=0)


def moment(field, forces, boundary, centerpoint=np.zeros(3)):
    r"""Return the sum of the moment vectors at the points of a boundary.

    ..  warning::
        Only 3D fields are supported.

    Parameters
    ----------
    field : felupe.FieldContainer
        The field container. The function uses only the first field of the container.
    forces : ndarray or scipy.sparse matrix
        The assembled vector of the internal forces. For example, use ``res.fun`` of
        :func:`~felupe.newtonraphson` or ``context.substep.fun`` in a
        :class:`~felupe.Plugin`.
    boundary : felupe.Boundary
        The boundary. The point ids ``boundary.points`` select the force vectors.
    centerpoint : array_like of shape (3,), optional
        The coordinates of the center point. This is not a point id. Default is the
        origin.

    Returns
    -------
    ndarray of shape (3,)
        The sum of the moment vectors at the points of the boundary.

    See Also
    --------
    felupe.tools.force : Return the sum of the force vectors at the points of a
        boundary.

    Notes
    -----
    The lever arms use the deformed configuration. The lever arm of a point is the
    point coordinates plus the displacements, minus the center point, see Eq.
    :eq:`tools-moment`.

    ..  math::
        :label: tools-moment

        \boldsymbol{M} = \sum_{a \in \mathcal{B}} \left( \boldsymbol{X}_a
            + \boldsymbol{u}_a - \boldsymbol{c} \right) \times \boldsymbol{f}_a

    Here, :math:`\mathcal{B}` is the set of point ids of the boundary,
    :math:`\boldsymbol{X}_a` are the point coordinates, :math:`\boldsymbol{u}_a` are the
    displacements, :math:`\boldsymbol{c}` is the center point and
    :math:`\boldsymbol{f}_a` are the force vectors.

    Examples
    --------
    Apply uniaxial tension to a cube. Then, calculate the reaction moment on the
    boundary ``"move"``.

    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> region = fem.RegionHexahedron(fem.Cube(n=3))
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, return_loadcase=True)
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)
    >>> res = fem.newtonraphson(items=[solid], **loadcase, verbose=0)
    >>>
    >>> M = fem.tools.moment(field, res.fun, boundaries["move"])
    >>> print(M[1:].round(3))
    [ 0.203 -0.203]

    Use a point on the deformed end face as the center point.

    >>> M = fem.tools.moment(
    ...     field, res.fun, boundaries["move"], centerpoint=[1.2, 0.5, 0.5]
    ... )
    >>> print(M[1:].round(3))
    [-0.011  0.011]

    """

    if issparse(forces):
        forces = forces.toarray()

    dim = field[0].dim
    centerpoint = np.asarray(centerpoint).reshape(1, -1)[:, :dim]

    displacements = field[0].values
    force = (np.split(forces, field.offsets)[0]).reshape(-1, dim)

    moments = cross(
        (field.region.mesh.points + displacements - centerpoint)[boundary.points].T,
        force[boundary.points].T,
    ).T

    return moments.sum(axis=0)


def curve(x, y, num=50):
    r"""Return the data points and an interpolated curve of given (x, y) data.

    Parameters
    ----------
    x : ndarray of shape (n,)
        The x-values. The array can have more values than ``y``. The function uses
        only the first ``len(y)`` values of ``x``.
    y : ndarray of shape (m,)
        The y-values, with ``2 <= m <= n``.
    num : int, optional
        The number of points of the interpolated curve (default is 50).

    Returns
    -------
    ndarray of shape (2, m)
        The data points. Row 0 has the x-values and row 1 has the y-values.
    ndarray of shape (2, num)
        The interpolated curve. Row 0 has the x-values and row 1 has the y-values.

    Notes
    -----
    The number of y-values sets the type of interpolation. Two values give a linear
    curve. Three values give a quadratic curve. Four or more values give a cubic
    curve.

    Examples
    --------
    The array of x-values has more values than the array of y-values.

    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> x = np.linspace(0, 1, 5)
    >>> y = np.array([0.0, 0.5, 0.8])
    >>> data, interpolated = fem.tools.curve(x, y, num=4)
    >>> data
    array([[0.  , 0.25, 0.5 ],
           [0.  , 0.5 , 0.8 ]])
    >>> interpolated
    array([[0.        , 0.16666667, 0.33333333, 0.5       ],
           [0.        , 0.35555556, 0.62222222, 0.8       ]])

    """

    kind = [None, "linear", "quadratic", "cubic"][min(len(y), 4) - 1]

    f = interp1d(x[: len(y)], y, kind=kind)

    xt = x[: len(y)]
    xx = np.linspace(xt[0], xt[-1], num=num)

    return np.array([xt, y]), np.array([xx, f(xx)])

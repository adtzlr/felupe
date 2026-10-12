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

import warnings
from math import factorial

import numpy as np
from numpy.polynomial import Polynomial
from scipy.sparse import coo_matrix, csr_matrix

from ._helpers import Assemble, Results


def _dof(mesh):
    "Return the global degrees of freedom of all points, shape (npoints, dim)."
    return np.arange(mesh.ndof).reshape(mesh.points.shape)


def _assemble_vector(dof, points, centerpoint, force, active):
    """Assemble the sparse vector of a center-point which is connected to points.

    Parameters
    ----------
    dof : (npoints, dim) ndarray of int
        The global degrees of freedom of all points.
    points : (n,) ndarray of int
        The indices of the connected points.
    centerpoint : int
        The index of the center-point.
    force : (n, dim) ndarray
        The forces of the connections, acting on the center-point.
    active : (n, dim) ndarray of bool
        A mask of the active components of the connections.

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled sparse vector of shape (ndof, 1).
    """

    axes = np.nonzero(active)[1]
    force = force[active]

    rows = np.concatenate([dof[points][active], dof[centerpoint][axes]])
    values = np.concatenate([-force, force])

    return csr_matrix((values, (rows, np.zeros_like(rows))), shape=(dof.size, 1))


def _assemble_matrix(dof, points, centerpoint, multiplier, active):
    """Assemble the sparse matrix of a center-point which is connected to points by
    springs with a given stiffness (multiplier) for all active components.

    Parameters
    ----------
    dof : (npoints, dim) ndarray of int
        The global degrees of freedom of all points.
    points : (n,) ndarray of int
        The indices of the connected points.
    centerpoint : int
        The index of the center-point.
    multiplier : float
        The stiffness of the connections.
    active : (n, dim) ndarray of bool
        A mask of the active components of the connections.

    Returns
    -------
    scipy.sparse.csr_matrix
        The assembled sparse matrix of shape (ndof, ndof).
    """

    axes = np.nonzero(active)[1]

    t = dof[points][active]
    c = dof[centerpoint][axes]
    k = np.full(len(t), float(multiplier))

    rows = np.concatenate([t, t, c, c])
    cols = np.concatenate([t, c, t, c])
    values = np.concatenate([k, -k, -k, k])

    return coo_matrix((values, (rows, cols)), shape=(dof.size, dof.size)).tocsr()


def _skew(w):
    "Return the skew-symmetric matrix [w]× of a vector with three components."
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


# skew-symmetric matrices [e_a]× of the base vectors, shape (3, 3, 3)
_SKEW_BASE = np.array([_skew(e) for e in np.eye(3)])

# power series of the coefficients alpha(s) = sin(t) / t and beta(s) = (1 - cos(t)) / s
# of the rotation matrix with s = t², along with their first and second derivatives
_POWER_SERIES = [
    Polynomial([(-1) ** k / factorial(2 * k + offset) for k in range(12)])
    for offset in [1, 2]
]
_SERIES = [[series.deriv(m) for m in range(3)] for series in _POWER_SERIES]


def _rotation_coefficients(s):
    r"""Return the coefficients :math:`\alpha(s) = \sin(t) / t` and
    :math:`\beta(s) = (1 - \cos(t)) / t^2` of the rotation matrix with :math:`s = t^2`,
    along with their first and second derivatives w.r.t. :math:`s`.

    Notes
    -----
    The closed-form expressions are undefined for :math:`s = 0`, which is the state at
    the beginning of each increment of the rotation. Further, their derivatives are
    differences of nearly equal numbers, divided by :math:`s` (or :math:`s^2`), i.e.
    they suffer from cancellation for small angles, e.g. a relative error of about
    1e-2 for :math:`\beta''` at :math:`s = 10^{-4}`. Hence, the Taylor series of the
    sine and the cosine (divided by :math:`t` and :math:`t^2`, with 12 terms) are used
    for :math:`s < 1`, which are accurate to machine precision there. Both branches
    agree within 1e-13 at :math:`s = 1`."""

    if s < 1.0:
        alpha, beta = [[term(s) for term in series] for series in _SERIES]

    else:
        t = np.sqrt(s)
        cos = np.cos(t)
        a = np.sin(t) / t
        b = (1 - cos) / s
        da = (cos - a) / (2 * s)
        db = (a - 2 * b) / (2 * s)
        alpha = [a, da, -(a + 6 * da) / (4 * s)]
        beta = [b, db, (da - 4 * db) / (2 * s)]

    return alpha, beta


def _rotation(w, derivatives=True):
    r"""Return the rotation matrix
    :math:`\boldsymbol{Q} = \exp([\boldsymbol{w}]_\times)` of a rotation vector
    :math:`\boldsymbol{w}` and (optionally) its first and second derivatives w.r.t. the
    rotation vector with shapes (3, 3), (3, 3, 3) and (3, 3, 3, 3), where the axes of
    the rotation vector come first.

    Notes
    -----
    The rotation matrix is given by Rodrigues' formula with the skew-symmetric matrix
    :math:`\boldsymbol{S} = [\boldsymbol{w}]_\times`, the coefficients
    :math:`\alpha(s)` and :math:`\beta(s)` and :math:`s = \boldsymbol{w} \cdot
    \boldsymbol{w}`.

    ..  math::

        \boldsymbol{Q} = \boldsymbol{1} + \alpha\ \boldsymbol{S}
            + \beta\ \boldsymbol{S}^2

    With :math:`\boldsymbol{E}_a = [\boldsymbol{e}_a]_\times`, the derivatives are

    ..  math::

        \frac{\partial \boldsymbol{Q}}{\partial w_a} &= 2 w_a \left(
            \alpha'\ \boldsymbol{S} + \beta'\ \boldsymbol{S}^2 \right)
            + \alpha\ \boldsymbol{E}_a
            + \beta \left( \boldsymbol{E}_a \boldsymbol{S}
            + \boldsymbol{S} \boldsymbol{E}_a \right)

        \frac{\partial^2 \boldsymbol{Q}}{\partial w_a\ \partial w_b} &=
            2 \delta_{ab} \left( \alpha'\ \boldsymbol{S}
            + \beta'\ \boldsymbol{S}^2 \right)
            + 4 w_a w_b \left( \alpha''\ \boldsymbol{S}
            + \beta''\ \boldsymbol{S}^2 \right)
            + 2 w_a\ \boldsymbol{T}_b + 2 w_b\ \boldsymbol{T}_a
            + \beta \left( \boldsymbol{E}_a \boldsymbol{E}_b
            + \boldsymbol{E}_b \boldsymbol{E}_a \right)

        \boldsymbol{T}_a &= \alpha'\ \boldsymbol{E}_a
            + \beta' \left( \boldsymbol{E}_a \boldsymbol{S}
            + \boldsymbol{S} \boldsymbol{E}_a \right)

    where :math:`(\bullet)'` denotes the derivative w.r.t. :math:`s`.
    """

    w = np.asarray(w, dtype=float)
    (a, da, dda), (b, db, ddb) = _rotation_coefficients(w @ w)

    S = _skew(w)
    SS = S @ S
    Q = np.eye(3) + a * S + b * SS

    if not derivatives:
        return Q

    E = _SKEW_BASE
    ES = E @ S + S @ E
    P = da * S + db * SS
    T = da * E + db * ES
    EE = E[:, None] @ E[None, :]

    dQ = 2 * w.reshape(3, 1, 1) * P + a * E + b * ES
    ddQ = (
        2 * np.eye(3).reshape(3, 3, 1, 1) * P
        + 4 * np.outer(w, w).reshape(3, 3, 1, 1) * (dda * S + ddb * SS)
        + 2 * w.reshape(3, 1, 1, 1) * T[None, :]
        + 2 * w.reshape(1, 3, 1, 1) * T[:, None]
        + b * (EE + EE.transpose([1, 0, 2, 3]))
    )

    return Q, dQ, ddQ


class MultiPointConstraint:
    r"""A Multi-point-constraint which connects a center-point to a list of points.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray
        An array with indices of points to be connected to the center-point.
    centerpoint : int
        The index of the centerpoint.
    skip : 3-tuple of bool, optional
        A tuple with boolean values for each axis to skip. If True, the respective axis
        is not connected. Not supported with a rotation-point. Default is
        (False, False, False).
    multiplier : float, optional
        A multiplier to penalize the relative displacements between the center-point and
        the points. Default is 1e3.
    rotationpoint : int or None, optional
        The index of an additional point, whose degrees of freedom are the rotations
        (in rad) of the points about the center-point, see the notes below. Its
        coordinates are not used, i.e. it may be located at the center-point. If None,
        the rotations are locked. Default is None.

    Attributes
    ----------
    results : Results
        The results of the last evaluation. With a rotation-point, these include the
        rotation matrix ``results.rotation`` with shape (3, 3), the moment
        ``results.moment`` with shape (3,) and the (converged) state variables
        ``results.statevars`` with the accumulated rotation vector and the rotation
        matrix of the last converged substep, shape (12,).

    Notes
    -----
    A :class:`~felupe.MultiPointConstraint` is supported as an item in a
    :class:`~felupe.Step`. It provides the assemble-methods
    :meth:`MultiPointConstraint.assemble.vector() <felupe.MultiPointConstraint.assemble.vector>`
    and :meth:`MultiPointConstraint.assemble.matrix() <felupe.MultiPointConstraint.assemble.matrix>`.

    The points :math:`\boldsymbol{x}_i` are connected to the center-point
    :math:`\boldsymbol{x}_c` by the penalty potential in Eq. :eq:`mpc-potential`, with
    the undeformed arms :math:`\boldsymbol{A}_i = \boldsymbol{X}_i - \boldsymbol{X}_c`
    and the rotation matrix :math:`\boldsymbol{R}`. Without a rotation-point, the
    rotation matrix is the identity, i.e. the points are moved by the displacements of
    the center-point and the rotations are locked.

    ..  math::
        :label: mpc-potential

        \Pi &= \sum_i \frac{k}{2}\ \boldsymbol{d}_i \cdot \boldsymbol{d}_i

        \boldsymbol{d}_i &= \left( \boldsymbol{x}_i - \boldsymbol{x}_c \right)
            - \boldsymbol{R}\ \boldsymbol{A}_i

    With a rotation-point, the rotation matrix is parametrized incrementally by the
    exponential map [1]_, see Eq. :eq:`mpc-rotation`, where the rotation-vector
    :math:`\boldsymbol{\psi}` are the values of the displacement field at the
    rotation-point. The rotation-vector
    :math:`\boldsymbol{\psi}_n` and the rotation matrix :math:`\boldsymbol{R}_n` of the
    last converged substep are state variables, which are updated after each converged
    substep.

    ..  math::
        :label: mpc-rotation

        \boldsymbol{R} &= \exp \left( [\Delta \boldsymbol{\psi}]_\times \right)\
            \boldsymbol{R}_n, \qquad
        \Delta \boldsymbol{\psi} = \boldsymbol{\psi} - \boldsymbol{\psi}_n

        \exp \left( [\boldsymbol{w}]_\times \right) &= \boldsymbol{1}
            + \frac{\sin \theta}{\theta}\ [\boldsymbol{w}]_\times
            + \frac{1 - \cos \theta}{\theta^2}\ [\boldsymbol{w}]_\times^2, \qquad
        \theta = \| \boldsymbol{w} \|

    Hence, the values of the rotation-point accumulate the (spatial) increments of the
    rotation. For a rotation about a fixed axis :math:`\boldsymbol{a}`, these are the
    rotation-vector :math:`\boldsymbol{\psi} = \varphi\ \boldsymbol{a}`, also for
    angles beyond :math:`2\pi`. For rotations about varying axes, the values depend on
    the path of the rotation and the rotation matrix is available as
    ``results.rotation``. The parametrization is singular only for increments
    :math:`\| \Delta \boldsymbol{\psi} \| = 2\pi`, see [2]_ for a discussion of
    vector-like parametrizations of finite rotations. However, the increments of the
    rotation must be smaller than :math:`\pi` (and should be much smaller), otherwise a
    revolution of the rotation-point may be skipped by the connected points.

    The moment of the forces of the constraint, which act on the points, w.r.t. the
    deformed center-point is given in Eq. :eq:`mpc-moment` and it is stored in
    ``results.moment``. The residuals of the rotation-point are the derivatives of the
    potential w.r.t. :math:`\boldsymbol{\psi}`, i.e. the moment transformed by the
    tangent operator :math:`\boldsymbol{J}` of the exponential map. These are equal to
    the moment for :math:`\Delta \boldsymbol{\psi} = \boldsymbol{0}` and differ by a
    term of the order of the increment :math:`\| \Delta \boldsymbol{\psi} \|`
    otherwise. For a rotation about a fixed axis, the axial components are equal.

    ..  math::
        :label: mpc-moment

        \boldsymbol{m} = \sum_i \left( \boldsymbol{x}_i - \boldsymbol{x}_c \right)
            \times \left( -k\ \boldsymbol{d}_i \right), \qquad
        \frac{\partial \Pi}{\partial \boldsymbol{\psi}} =
            \boldsymbol{J}(\Delta \boldsymbol{\psi})^T\ \boldsymbol{m}

    For 2d-meshes, the first component of the rotation-point is the angle of the
    rotation about the :math:`z`-axis and the second component is locked by the
    penalty multiplier. If the rotation-point is located at the center-point, both
    points must be selected by a mask (or by their indices), e.g. for a
    :class:`~felupe.Boundary`, because a selection by the coordinates would include both
    points.

    ..  note::

        A :class:`~felupe.PointLoad` at the rotation-point is work-conjugate to the
        rotation-vector :math:`\boldsymbol{\psi}`. Hence, the moment
        :math:`\boldsymbol{m}` differs from the applied load by a term of the order of
        the increment of the rotation, except for the axial component of a rotation
        about a fixed axis.

    Examples
    --------
    This example shows how to use a :class:`~felupe.MultiPointConstraint`. An
    additional center-point is added to a mesh.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> mesh.add_points([2.0, 0.5, 0.5])
        >>>
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>>
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> solid = fem.SolidBody(umat=umat, field=field)

    A :class:`~felupe.MultiPointConstraint` defines the multi-point constraint which
    connects the displacement degrees of freedom of the center-point with the dofs of
    points located at :math:`x=1`.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> mpc = fem.MultiPointConstraint(
        ...     field=field,
        ...     points=np.arange(mesh.npoints)[mesh.x == 1],
        ...     centerpoint=-1,
        ... )
        >>>
        >>> plotter = pv.Plotter()
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[mpc.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[mpc.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> mesh.plot(plotter=mpc.plot(plotter=plotter)).show()

    The mesh is fixed on the left end face and a ramped :class:`~felupe.PointLoad` is
    applied on the center-point of the :class:`~felupe.MultiPointConstraint`. All items
    are added to a :class:`~felupe.Step` and a :class:`~felupe.Job` is evaluated.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {"fixed": fem.Boundary(displacement, fx=0)}
        >>> load = fem.PointLoad(field, points=[-1])
        >>> table = fem.math.linsteps([0, 1], num=5, axis=0, axes=3)
        >>>
        >>> step = fem.Step(
        ...     [solid, mpc, load], boundaries=boundaries, ramp={load: table}
        ... )
        >>> job = fem.Job([step]).evaluate()

    A view on the deformed mesh including the :class:`~felupe.MultiPointConstraint` is
    plotted.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> plotter = pv.Plotter()
        >>>
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[mpc.points] + displacement.values[mpc.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[mpc.centerpoint]] + displacement.values[[mpc.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> field.plot("Displacement", component=None, plotter=mpc.plot(plotter=plotter)).show()

    With a rotation-point, the points are also rotated about the center-point. A
    center-point and a rotation-point are added at the center of the right end face of
    a beam. Both points are excluded from the connected points of the end face.

    ..  pyvista-plot::
        :context:

        >>> mesh = fem.Cube(b=(5, 1, 1), n=(11, 3, 3))
        >>> mesh.add_points([[5.0, 0.5, 0.5], [5.0, 0.5, 0.5]])
        >>>
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
        >>>
        >>> face = np.isclose(mesh.x, 5)
        >>> face[-2:] = False
        >>> mpc = fem.MultiPointConstraint(
        ...     field, points=face, centerpoint=-2, rotationpoint=-1
        ... )

    The left end face and the center-point are fixed. The end face is twisted by a full
    revolution about the :math:`x`-axis, i.e. the first component of the rotation-point
    is ramped, while the rotations about the other axes are free. The center-point and
    the rotation-point are located at the same coordinates, hence they are selected by
    masks.

    ..  pyvista-plot::
        :context:

        >>> centerpoint = np.zeros(mesh.npoints, dtype=bool)
        >>> rotationpoint = np.zeros(mesh.npoints, dtype=bool)
        >>> centerpoint[-2] = rotationpoint[-1] = True
        >>>
        >>> boundaries = {
        ...     "fixed": fem.Boundary(displacement, fx=0),
        ...     "center": fem.Boundary(displacement, mask=centerpoint),
        ...     "twist": fem.Boundary(displacement, mask=rotationpoint, skip=(0, 1, 1)),
        ... }
        >>> table = fem.math.linsteps([0, 2 * np.pi], num=16)
        >>> step = fem.Step(
        ...     [solid, mpc], boundaries=boundaries, ramp={boundaries["twist"]: table}
        ... )
        >>> job = fem.Job([step]).evaluate()

    After a full revolution, the rotation matrix is the identity, while the accumulated
    rotation of the rotation-point is :math:`2\pi`. The moment w.r.t. the center-point
    is available in the results.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> np.allclose(mpc.results.rotation, np.eye(3))
        True

        >>> np.allclose(displacement.values[-1], [2 * np.pi, 0, 0])
        True

        >>> plotter = mpc.plot()
        >>> solid.plot("Principal Values of Cauchy Stress", plotter=plotter).show()

    References
    ----------
    .. [1] J. C. Simo and L. Vu-Quoc, "A three-dimensional finite-strain rod model.
       Part II: Computational aspects", Computer Methods in Applied Mechanics and
       Engineering, vol. 58, no. 1, pp. 79–116, 1986.

    .. [2] A. Ibrahimbegović, F. Frey and I. Kožar, "Computational aspects of
       vector-like parametrization of three-dimensional finite rotations",
       International Journal for Numerical Methods in Engineering, vol. 38, no. 21,
       pp. 3653–3673, 1995.

    See Also
    --------
    felupe.MultiPointContact : A frictionless point-to-rigid (wall) contact.

    """

    def __init__(
        self,
        field,
        points,
        centerpoint,
        skip=(False, False, False),
        multiplier=1e3,
        rotationpoint=None,
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.array(points)
        self.centerpoint = centerpoint
        self.rotationpoint = rotationpoint
        self.mask = ~np.array(skip, dtype=bool)[: self.mesh.dim]
        self.axes = np.arange(self.mesh.dim)[self.mask]
        self.multiplier = multiplier

        if len(self.points) == 0:
            self.points = self.points.astype(int)

        if self.points.dtype == bool:
            self.points = np.where(self.points)[0]

        if self.centerpoint < 0:
            self.centerpoint = self.mesh.npoints + self.centerpoint

        if self.centerpoint in self.mesh.points_without_cells:
            self.mesh.points_without_cells = self.mesh.points_without_cells[
                self.mesh.points_without_cells != self.centerpoint
            ]

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

        if self.rotationpoint is not None:
            self._init_rotation()

    def _init_rotation(self):
        "Initialize the rotation-point and the state variables of the rotation."

        dim = self.mesh.dim

        if dim not in [2, 3]:
            raise ValueError("A rotation-point requires a 2d- or a 3d-mesh.")

        if not np.all(self.mask):
            raise ValueError("Skipped axes are not supported with a rotation-point.")

        if self.rotationpoint < 0:
            self.rotationpoint = self.mesh.npoints + self.rotationpoint

        if self.rotationpoint == self.centerpoint or self.rotationpoint in self.points:
            raise ValueError(
                "The rotation-point must differ from the center-point and the points."
            )

        if self.rotationpoint in self.mesh.points_without_cells:
            self.mesh.points_without_cells = self.mesh.points_without_cells[
                self.mesh.points_without_cells != self.rotationpoint
            ]

        # components of the rotation-vector of the values of the rotation-point (rows),
        # the second value of 2d-meshes is not used (rotation about the z-axis only)
        self._rotation_axes = np.eye(3)

        if dim == 2:
            self._rotation_axes = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

        # initial rotation-vector and rotation matrix as state variables
        values = self.field.fields[0].values
        psi = values[self.rotationpoint] @ self._rotation_axes
        rotation = _rotation(psi, derivatives=False)

        self.results.statevars = np.concatenate([psi, rotation.ravel()]).astype(
            values.dtype
        )
        self.results.rotation = rotation
        self.results.moment = np.zeros(3)

    def checkpoint(self):
        """Return a checkpoint of the state variables of the rotation.

        Returns
        -------
        dict
            A dict with a copy of the state variables (or None without a
            rotation-point).

        Notes
        -----
        The state variables are only updated after a converged substep. The field is
        not included in the checkpoint, see e.g.
        :meth:`SolidBody.checkpoint() <felupe.SolidBody.checkpoint>`.

        See Also
        --------
        felupe.MultiPointConstraint.restore : Restore a checkpoint of the constraint.
        felupe.CutbackPlugin : A cutback of the increment of failed substeps.
        """

        statevars = self.results.statevars

        if statevars is not None:
            statevars = statevars.copy()

        return {"results.statevars": statevars}

    def restore(self, checkpoint):
        """Restore a checkpoint of the state variables of the rotation inplace.

        Parameters
        ----------
        checkpoint : dict
            A dict with checkpoint arrays.

        See Also
        --------
        felupe.MultiPointConstraint.checkpoint : Return a checkpoint of the constraint.
        """

        statevars = checkpoint["results.statevars"]

        if statevars is not None:
            statevars = statevars.copy()

        self.results.statevars = statevars
        self.results._statevars = None

        # reset force and stiffness
        self.results.force = None
        self.results.stiffness = None

    def plot(self, plotter=None, color="black", deformed=True, **kwargs):
        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()

        # get undeformed points
        x = self.mesh.points

        if deformed:  # get deformed points
            x = x + self.field[0].values

        x = np.pad(x, ((0, 0), (0, 3 - x.shape[1])))
        pointa = x[self.centerpoint]

        for pointb in x[self.points]:
            plotter.add_mesh(pv.Line(pointa, pointb), color=color, **kwargs)

        return plotter

    def _vector(self, field=None, parallel=False):
        "Calculate vector of residuals with RBE2 contributions."

        if field is not None:
            self.field = field

        if self.rotationpoint is not None:
            self.results.force = self._vector_rotation()
            return self.results.force

        u = self.field.fields[0].values
        force = self.multiplier * (-u[self.points] + u[self.centerpoint])
        active = np.broadcast_to(self.mask, force.shape)

        self.results.force = _assemble_vector(
            _dof(self.mesh), self.points, self.centerpoint, force, active
        )
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate stiffness with RBE2 contributions."

        if field is not None:
            self.field = field

        active = np.broadcast_to(self.mask, (len(self.points), self.mesh.dim))

        self.results.stiffness = _assemble_matrix(
            _dof(self.mesh), self.points, self.centerpoint, self.multiplier, active
        )

        if self.rotationpoint is not None:
            self.results.stiffness += self._matrix_rotation()

        return self.results.stiffness

    def _kinematics(self):
        """Evaluate the rotation-vector, the rotation matrix of the increment and its
        derivatives, the rotated undeformed arms, the deformed arms and the gaps of the
        points, all padded to three components."""

        dim = self.mesh.dim
        u = self.field.fields[0].values
        X = self.mesh.points

        # converged state variables (the rotation-vector and the rotation matrix)
        statevars = self.results.statevars.astype(float)
        psi_n, rotation_n = statevars[:3], statevars[3:].reshape(3, 3)

        # rotation-vector and rotation matrix of the increment (with derivatives)
        psi = u[self.rotationpoint] @ self._rotation_axes
        increment = _rotation(psi - psi_n)

        pad = ((0, 0), (0, 3 - dim))
        dX = np.pad(X[self.points] - X[self.centerpoint], pad)
        du = np.pad(u[self.points] - u[self.centerpoint], pad)

        arms = dX @ rotation_n.T  # undeformed arms, rotated by the converged rotation
        dx = dX + du  # deformed arms
        gap = dx - arms @ increment[0].T

        return psi, rotation_n, increment, arms, dx, gap

    def _vector_rotation(self):
        "Calculate the vector of residuals with a rotation-point."

        dim = self.mesh.dim
        k = self.multiplier
        u = self.field.fields[0].values
        dof = _dof(self.mesh)

        psi, rotation_n, (Q, dQ, _), arms, dx, gap = self._kinematics()
        force = k * gap

        # residuals of the rotation-point (the unused value of 2d-meshes is locked)
        force_rotation = self._rotation_axes @ np.einsum(
            "aij,ij->a", dQ, -force.T @ arms
        )
        unused = ~np.any(self._rotation_axes, axis=1)
        force_rotation += k * unused * u[self.rotationpoint]

        rows = np.concatenate(
            [dof[self.points].ravel(), dof[self.centerpoint], dof[self.rotationpoint]]
        )
        values = np.concatenate(
            [force[:, :dim].ravel(), -force[:, :dim].sum(axis=0), force_rotation]
        )

        # rotation matrix, moment w.r.t. the center-point and trial state variables
        rotation = Q @ rotation_n
        self.results.rotation = rotation
        self.results.moment = np.cross(dx, -force).sum(axis=0)
        self.results._statevars = np.concatenate([psi, rotation.ravel()]).astype(
            u.dtype
        )

        return csr_matrix(
            (values.astype(u.dtype), (rows, np.zeros_like(rows))),
            shape=(dof.size, 1),
        )

    def _matrix_rotation(self):
        """Calculate the stiffness of the rotation-point and its coupling to the points
        and the center-point (without the stiffness of the translational springs)."""

        dim = self.mesh.dim
        k = self.multiplier
        dof = _dof(self.mesh)
        npoints = len(self.points)

        _, _, (_, dQ, ddQ), arms, dx, _ = self._kinematics()
        axes = self._rotation_axes

        # derivatives of the rotated arms w.r.t. the values of the rotation-point,
        # shape (point, component, value)
        darms = np.einsum("aij,nj->nia", dQ, arms)[:, :dim] @ axes.T

        K_points = -k * darms
        K_center = k * darms.sum(axis=0)
        K_rotation = axes @ (-k * np.einsum("abij,ij->ab", ddQ, dx.T @ arms)) @ axes.T

        # lock the unused value of 2d-meshes
        K_rotation += k * np.diag(~np.any(axes, axis=1))

        p = dof[self.points]
        c = dof[self.centerpoint]
        r = dof[self.rotationpoint]

        rows_points = np.broadcast_to(p.reshape(npoints, dim, 1), K_points.shape)
        cols_points = np.broadcast_to(r.reshape(1, 1, dim), K_points.shape)
        rows_center = np.broadcast_to(c.reshape(dim, 1), K_center.shape)
        cols_center = np.broadcast_to(r.reshape(1, dim), K_center.shape)

        rows = [rows_points, cols_points, rows_center, cols_center, cols_center.T]
        cols = [cols_points, rows_points, cols_center, rows_center, cols_center]
        values = [K_points, K_points, K_center, K_center, K_rotation]

        return coo_matrix(
            (
                np.concatenate([v.ravel() for v in values]),
                (
                    np.concatenate([i.ravel() for i in rows]),
                    np.concatenate([j.ravel() for j in cols]),
                ),
            ),
            shape=(dof.size, dof.size),
        ).tocsr()


class MultiPointContact:
    """A frictionless point-to-rigid (wall) contact which connects a center-point to a
    list of points.

    Parameters
    ----------
    field : FieldContainer
        A field container with the displacement field as first field.
    points : (n,) ndarray
        An array with indices of points to be connected to the center-point.
    centerpoint : int
        The index of the centerpoint.
    skip : 3-tuple of bool, optional
        A tuple with boolean values for each axis to skip. If True, the respective axis
        is not connected. Default is (False, False, False).
    multiplier : float, optional
        A multiplier to penalize the relative displacements between the center-point and
        the points in contact. Default is 1e6.

    Notes
    -----
    A :class:`~felupe.MultiPointContact` is supported as an item in a
    :class:`~felupe.Step`. It provides the assemble-methods
    :meth:`MultiPointContact.assemble.vector() <felupe.MultiPointContact.assemble.vector>`
    and :meth:`MultiPointContact.assemble.matrix() <felupe.MultiPointContact.assemble.matrix>`.

    Examples
    --------
    This example shows how to use a :class:`~felupe.MultiPointContact`. An additional
    center-point is added to a mesh.

    ..  pyvista-plot::
        :context:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> mesh.add_points([2.0, 0.5, 0.5])
        >>>
        >>> # prevent the field-values at the center-point to be treated as dof0
        >>> mesh.points_without_cells = mesh.points_without_cells[:-1]
        >>>
        >>> region = fem.RegionHexahedron(mesh)
        >>> displacement = fem.Field(region, dim=3)
        >>> field = fem.FieldContainer([displacement])
        >>>
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> solid = fem.SolidBody(umat=umat, field=field)

    A :class:`~felupe.MultiPointContact` defines the multi-point contact which
    connects the displacement degrees of freedom of the center-point with the dofs of
    points located at :math:`x=1` if they are in contact. Only the :math:`x`-component
    is considered in this example.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> contact = fem.MultiPointContact(
        ...     field=field,
        ...     points=np.arange(mesh.npoints)[mesh.x == 1],
        ...     centerpoint=-1,
        ...     skip=(False, True, True)
        ... )
        >>>
        >>> plotter = pv.Plotter()
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[contact.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[contact.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> mesh.plot(plotter=contact.plot(plotter=plotter)).show()

    The mesh is fixed on the left end face and a ramped :class:`~felupe.Boundary` is
    applied on the center-point of the :class:`~felupe.MultiPointContact`. All items
    are added to a :class:`~felupe.Step` and a :class:`~felupe.Job` is evaluated.

    ..  pyvista-plot::
        :context:

        >>> boundaries = {
        ...     "fixed": fem.Boundary(displacement, fx=0),
        ...     "control": fem.Boundary(displacement, fx=2, skip=(1, 0, 0)),
        ...     "move": fem.Boundary(displacement, fx=2, skip=(0, 1, 1)),
        ... }
        >>> table = fem.math.linsteps([0, -1, -1.5], num=5)
        >>> step = fem.Step(
        ...     [solid, contact],
        ...     boundaries=boundaries,
        ...     ramp={boundaries["move"]: table},
        ... )
        >>> job = fem.Job([step]).evaluate()

    A view on the deformed mesh including the :class:`~felupe.MultiPointContact` is
    plotted.

    ..  pyvista-plot::
        :context:
        :force_static:

        >>> plotter = pv.Plotter()
        >>>
        >>> actor_1 = plotter.add_points(
        ...     mesh.points[contact.points] + displacement.values[contact.points],
        ...     point_size=16,
        ...     color="red",
        ... )
        >>> actor_2 = plotter.add_points(
        ...     mesh.points[[contact.centerpoint]] + displacement.values[[contact.centerpoint]],
        ...     point_size=16,
        ...     color="green",
        ... )
        >>> field.plot("Displacement", component=None, plotter=contact.plot(plotter=plotter)).show()

    See Also
    --------
    felupe.MultiPointConstraint : A Multi-point-constraint which connects a center-point
        to a list of points.

    """

    def __init__(
        self, field, points, centerpoint, skip=(False, False, False), multiplier=1e6
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.array(points)
        self.centerpoint = centerpoint
        self.mask = ~np.array(skip, dtype=bool)[: self.mesh.dim]
        self.axes = np.arange(self.mesh.dim)[self.mask]
        self.multiplier = multiplier

        if len(self.points) == 0:
            self.points = self.points.astype(int)

        if self.points.dtype == bool:
            self.points = np.where(self.points)[0]

        if self.centerpoint < 0:
            self.centerpoint = self.mesh.npoints + self.centerpoint

        if self.centerpoint in self.mesh.points_without_cells:
            self.mesh.points_without_cells = self.mesh.points_without_cells[
                self.mesh.points_without_cells != self.centerpoint
            ]

        self.results = Results(stress=False, elasticity=False)
        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)

    def plot(
        self,
        plotter=None,
        offset=0,
        show_edges=True,
        color="black",
        opacity=0.5,
        deformed=True,
        **kwargs,
    ):
        import pyvista as pv

        if plotter is None:
            plotter = pv.Plotter()

        # get edge lengths of undeformed enclosing box
        x = self.mesh.points

        if deformed:  # get deformed points
            x = x + self.field[0].values

        edges = np.diag((x.max(axis=0) - x.min(axis=0))) + x.min(axis=0)

        # plot a line or a rectangle for each active contact plane
        for ax in self.axes:
            # fill the point values of the normal axis with the centerpoint values
            points = edges.copy()
            points[:, ax] = x[self.centerpoint, ax] + offset

            # scale the line or rectangle at the origin
            origin = points.mean(axis=0)
            points = (points - origin) * 1.05 + origin

            # plot a line or a rectangle
            if len(points) == 3:
                plotter.add_mesh(
                    pv.Rectangle(points), color=color, opacity=opacity, **kwargs
                )
            else:
                points = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
                plotter.add_mesh(
                    pv.Line(*points), color=color, opacity=opacity, **kwargs
                )

        return plotter

    def _contact(self):
        """Return the gaps between the center-point and the points and a mask of the
        components in contact, i.e. with changed signs of the gaps."""

        u = self.field.fields[0].values

        Xc = self.mesh.points[self.centerpoint]
        Xt = self.mesh.points[self.points]

        xc = u[self.centerpoint] + Xc
        xt = u[self.points] + Xt

        gap = -xt + xc
        active = (np.sign(-Xt + Xc) != np.sign(gap)) & self.mask

        return gap, active

    def _vector(self, field=None, parallel=False):
        "Calculate vector of residuals with RBE2 contributions."

        if field is not None:
            self.field = field

        gap, active = self._contact()

        self.results.force = _assemble_vector(
            _dof(self.mesh),
            self.points,
            self.centerpoint,
            self.multiplier * gap,
            active,
        )
        return self.results.force

    def _matrix(self, field=None, parallel=False):
        "Calculate stiffness with RBE2 contributions."

        if field is not None:
            self.field = field

        _, active = self._contact()

        self.results.stiffness = _assemble_matrix(
            _dof(self.mesh), self.points, self.centerpoint, self.multiplier, active
        )
        return self.results.stiffness

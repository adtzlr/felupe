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

import numpy as np
from scipy.sparse.linalg import spsolve

from ..math import values


def partition(v, K, dof1, dof0, r=None):
    """Perform partitioning of field values (unknowns), (stiffness) matrix
    and (residuals) vector with given lists of active (dof1) and
    prescribed degrees of freedom (dof0).

    Parameters
    ----------
    v : FieldContainer
        A field container with the fields. It is used to extract and concatenate the
        1d-array of unknows.
    K : scipy.sparse.spmatrix
        A two-dimensional sparse (stiffness) matrix.
    dof1 : list of int
        List of active degrees of freedom (zero-indexed).
    dof0 : list of int
        List of prescribed degrees of freedom (zero-indexed).
    r : 1d-array or None, optional
        The 1d-vector of residuals. If None, the residuals are set to zero. Default is
        None.

    Returns
    -------
    u : 1d-array
        The concatenated full 1d-array of unknowns, extracted from the field container.
    u0 : 1d-array
        The prescribed unknowns.
    K11 : scipy.sparse.spmatrix
        The active-active block of the two-dimensional sparse (stiffness) matrix.
    K10 : scipy.sparse.spmatrix
        The active-prescribed block of the two-dimensional sparse (stiffness) matrix.
    dof1 : list of int
        List of active degrees of freedom (zero-indexed).
    dof0 : list of int
        List of prescribed degrees of freedom (zero-indexed).
    r1 : 1d-array
        The 1d-vector of active residuals.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=3)
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldPlaneStrain(region, dim=2).as_container()
        >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> solid = fem.SolidBody(umat, field)
        >>>
        >>> K = solid.assemble.matrix()
        >>> r = solid.assemble.vector()
        >>>
        >>> dof0 = loadcase["dof0"]
        >>> dof1 = loadcase["dof1"]
        >>> ext0 = loadcase["ext0"]
        >>>
        >>> system = fem.solve.partition(field, K, dof1, dof0, r)

    See Also
    --------
    felupe.solve.solve : Linear solution of equation system.
    """

    # extract values
    u = values(v)

    # check if residuals vector is passed
    if r is None:
        r1 = None
    else:
        # partition residuals vector
        r1 = r[dof1]

    # prescribed dofs of unknowns
    u0 = u.ravel()[dof0]

    # partition (stiffness) matrix
    K11 = K[dof1, :][:, dof1]
    K10 = K[dof1, :][:, dof0]

    return u, u0, K11, K10, dof1, dof0, r1


def solve(u, u0, K11, K10, dof1, dof0, r1=None, ext0=None, solver=spsolve):
    r"""Linear solution of equation system with optional given values of
    unknowns at prescribed deegrees of freedom.

    Notes
    -----

    ..  math::

        \boldsymbol{K}_{11}\ d\boldsymbol{u}_1 =
            -\boldsymbol{r}_1 - \boldsymbol{K}_{10} (
                \boldsymbol{u}_{0,\text{ext}} - \boldsymbol{u}_0
            )

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=3)
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldPlaneStrain(region, dim=2).as_container()
        >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> solid = fem.SolidBody(umat, field)
        >>>
        >>> K = solid.assemble.matrix()
        >>> r = solid.assemble.vector()
        >>>
        >>> dof0 = loadcase["dof0"]
        >>> dof1 = loadcase["dof1"]
        >>> ext0 = loadcase["ext0"]
        >>>
        >>> system = fem.solve.partition(field, K, dof1, dof0, r)
        >>> field += fem.solve.solve(*system, ext0=ext0)

    See Also
    --------
    felupe.solve.partition : Perform a partitioning into active and prescribed dof.
    """

    # init active residuals
    if r1 is None:
        r1 = np.zeros(len(dof1))

    # init external displacements
    if ext0 is None:
        ext0 = 0
        # init inactive dofs of residuals
        dr0 = np.zeros(len(dof1))
    else:
        # evaluate inactive dofs of residuals
        dr0 = K10.dot(ext0 - u0)

    # solve linear system (active dofs)
    du1 = solver(K11, -r1 - dr0.reshape(*r1.shape))

    # full solution
    du = np.empty(u.size)
    du[dof1] = du1
    du[dof0] = ext0 - u0

    # reshape solution to shape of input
    return du.reshape(*u.shape)

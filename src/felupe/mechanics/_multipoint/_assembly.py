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
from scipy.sparse import coo_matrix, csr_matrix


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

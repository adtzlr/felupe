# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from ._boundary import Boundary


def get_dof0(field, bounds):
    "Extract prescribed degrees of freedom."

    # get mesh from field and obtain field-dimension
    mesh = field.region.mesh
    dim = field.dim

    # check if there are points without connected cells in the mesh
    # and add them to the list of prescribed dofs
    # e.g. these are points [2,6,7]
    #
    #   ( [[2,2,2], )   [[0,1,2],   [[ 6, 7, 8],
    # 3*(  [6,6,6], ) +  [0,1,2], =  [18,19,20],
    #   (  [7,7,7]] )    [0,1,2]]    [21,22,23]]
    #
    # fixmissing = [6, 7, 8, 18, 19, 29, 21, 22, 23]
    fixmissing = dim * np.tile(mesh.points_without_cells, (dim, 1)).T + np.arange(dim)

    # obtain prescribed dofs from boundaries
    dof0_bounds = np.concatenate([b.dof for b in bounds.values()])

    # combine all prescribed dofs and remove repeated itmes if there are any
    return np.unique(np.append(fixmissing.ravel(), dof0_bounds))


def get_dof1(field, bounds, dof0):
    "Extract active (non-prescribed) degrees of freedom."

    # obtain all dofs from the field
    dof = field.indices.dof

    # init a mask for the selection of active dofs
    mask = np.ones_like(dof.ravel(), dtype=bool)

    # set mask items for prescribed dofs (dof0) to False
    mask[dof0] = False

    # make the dof list 1d and mask active dofs
    return dof.ravel()[mask]


def partition(field, bounds):
    "Partition dof-list into prescribed (dof0) and active (dof1) parts."

    fields = field.fields
    offsets = field.offsets

    # list of boundaries, partitioned by fields
    boundaries = [
        {key: value for key, value in bounds.items() if value.field == f}
        for f in fields
    ]

    # fix fields without boundaries (need at least one boundary per field)
    boundaries = [
        {"__empty__": Boundary(f)} if not b else b for f, b in zip(fields, boundaries)
    ]

    # partition degrees of freedom for each field
    dofs0 = [get_dof0(f, b) for f, b in zip(fields, boundaries)]
    dofs1 = [get_dof1(f, b, dof0=i) for f, b, i in zip(fields, boundaries, dofs0)]

    # concatenate degrees of freedom
    dof0 = np.concatenate(
        [dof0 + offset for dof0, offset in zip(dofs0, np.insert(offsets, 0, 0))]
    )
    dof1 = np.concatenate(
        [dof1 + offset for dof1, offset in zip(dofs1, np.insert(offsets, 0, 0))]
    )

    return dof0, dof1


def apply(field, bounds, dof0=None):
    """Apply prescribed values for a list of boundaries
    and return all (default) or only the prescribed components
    of the ``field`` based on the keyword ``dof0``."""

    # check if a mixed-field is passed
    u = np.concatenate([f.values.ravel() for f in field.fields])
    offsets = np.insert(field.offsets, 0, 0)

    for b in bounds.values():

        # get offset for field-dof of current boundary
        offset = offsets[[b.field == f for f in field.fields]]

        # set prescribed values
        u.ravel()[b.dof + offset] = b.value

    if dof0 is None:
        return u
    else:
        return u.ravel()[dof0]

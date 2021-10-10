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

from .._field import FieldMixed


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

    # if a tuple is passed it is assumed that the first
    # field is associated to the boundaries
    if isinstance(field, FieldMixed):
        f = field.fields[0]
        extend_dof1 = True
    else:
        f = field
        extend_dof1 = False

    dof0 = get_dof0(f, bounds)
    dof1 = get_dof1(f, bounds, dof0=dof0)

    # extend active dofs with dofs from additional fields
    if extend_dof1:
        dof0, dof1, offsets = extend(field, dof0, dof1)
        return dof0, dof1, offsets
    else:
        return dof0, dof1


def extend(field, dof0, dof1):
    "Extend partitioned dof-lists dof0 and dof1."

    # get sizes of fields and calculate offsets
    fieldsizes = [f.indices.dof.size for f in field.fields]
    offsets = np.cumsum(fieldsizes)

    # init extended dof0, dof1 arrays
    dof0_xt = dof0.copy()
    dof1_xt = dof1.copy()

    # loop over fields starting from the second one
    for fld, offset, fieldsize in zip(field.fields[1:], offsets[:-1], fieldsizes[1:]):

        # obtain the mesh and the dimension from the current field
        mesh = fld.region.mesh
        dim = fld.dim

        # check if there are points without/with connected cells in the mesh
        # and add them to the list of prescribed/active dofs
        # e.g. these are mesh.points_without_cells = [2,6,7]
        #
        #              ( [[2,2,2], )   [[0,1,2],   [[ 6, 7, 8],
        # dof0_add = 3*(  [6,6,6], ) +  [0,1,2], =  [18,19,20],
        #              (  [7,7,7]] )    [0,1,2]]    [21,22,23]]
        #
        dof0_add = (
            offset
            + dim * np.tile(mesh.points_without_cells, (dim, 1)).T
            + np.arange(dim)
        )
        dof1_add = (
            offset + dim * np.tile(mesh.points_with_cells, (dim, 1)).T + np.arange(dim)
        )

        dof0_xt = np.append(dof0_xt, dof0_add.ravel())
        dof1_xt = np.append(dof1_xt, dof1_add.ravel())

    return dof0_xt, dof1_xt, offsets[:-1]


def apply(v, bounds, dof0=None):
    """Apply prescribed values for a list of boundaries
    and return all (default) or only the prescribed components
    of the input array 'v' based on the keyword 'dof0'."""

    u = v.values.copy()

    for b in bounds.values():
        u.ravel()[b.dof] = b.value

    if dof0 is None:
        return u
    else:
        # check if dof0 has entries beyond the size of u
        # this is the case for meshes with points that are
        # not connected to cells
        u0ext = u.ravel()[dof0[dof0 < u.size]]

        # pad (=extend) u0ext to the full dimension of prescribed dofs
        # and fill padded values with zeros
        u0ext_padded = np.pad(u0ext, (0, len(dof0) - len(u0ext)))
        return u0ext_padded

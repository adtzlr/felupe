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
from types import SimpleNamespace


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


def get_dof1(field, bounds, dof0=None):
    "Extract active (non-prescribed) degrees of freedom."

    # obtain all dofs from the field
    dof = field.indices.dof

    # if function argument `dof0` is None call `get_dof0()`
    if dof0 is None:
        dof0 = get_dof0(field, bounds)

    # init a mask for the selection of active dofs
    mask = np.ones_like(dof.ravel(), dtype=bool)

    # set mask items for prescribed dofs (dof0) to False
    mask[dof0] = False

    # make the dof list 1d and mask active dofs
    return dof.ravel()[mask]


# convenient alias names
find = SimpleNamespace()
find.dof0 = get_dof0
find.dof1 = get_dof1


def partition(fields, bounds):
    "Partition dof-list into prescribed (dof0) and active (dof1) parts."

    # if a tuple is passed it is assumed that the first
    # field is associated to the boundaries
    if isinstance(fields, tuple):
        f = fields[0]
        extend_dof1 = True
    else:
        f = fields
        extend_dof1 = False

    dof0 = get_dof0(f, bounds)
    dof1 = get_dof1(f, bounds, dof0=dof0)

    # extend active dofs with dofs from additional fields
    if extend_dof1:
        dof0, dof1, offsets = extend(fields, dof0, dof1)
        return dof0, dof1, offsets
    else:
        return dof0, dof1


def extend(fields, dof0, dof1):
    "Extend partitioned dof-lists dof0 and dof1."

    # get sizes of fields and calculate offsets
    fieldsizes = [f.indices.dof.size for f in fields]
    offsets = np.cumsum(fieldsizes)

    # init extended dof0, dof1 arrays
    dof0_xt = dof0.copy()
    dof1_xt = dof1.copy()

    # loop over fields starting from the second one
    for field, offset, fieldsize in zip(fields[1:], offsets[:-1], fieldsizes[1:]):

        # obtain the mesh and the dimension from the current field
        mesh = field.region.mesh
        dim = field.dim

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


def symmetry(field, axes=(True, True, True), x=0, y=0, z=0, bounds=None):
    "Create symmetry boundary conditions."

    # obtain the mesh from the field
    # mesh = field.region.mesh

    # convert axes to array and slice by mesh dimension
    enforce = np.array(axes).astype(bool)[: field.dim]

    # create search functions for x,y,z - axes
    fx = lambda v: np.isclose(v, x)
    fy = lambda v: np.isclose(v, y)
    fz = lambda v: np.isclose(v, z)

    # invert boolean identity matrix and use its rows
    # for the skip argument (a symmetry condition on
    # axis "z" fixes all displacements u_z=0 but keeps
    # in-plane displacements active)
    skipax = ~np.eye(3).astype(bool)
    kwarglist = [
        {"fx": fx, "skip": skipax[0][: field.dim]},
        {"fy": fy, "skip": skipax[1][: field.dim]},
        {"fz": fz, "skip": skipax[2][: field.dim]},
    ]

    if bounds is None:
        bounds = {}
    labels = ["symx", "symy", "symz"]

    # loop over symmetry conditions and add them to a new dict
    for a, (symaxis, kwargs) in enumerate(zip(enforce, kwarglist[: field.dim])):
        if symaxis:
            bounds[labels[a]] = Boundary(field, **kwargs)

    return bounds


class Boundary:
    "A Boundary as a collection of prescribed dofs (coordinate components of a field at points of a mesh)."

    def __init__(
        self,
        field,
        name="default",
        fx=lambda v: v == np.nan,
        fy=lambda v: v == np.nan,
        fz=lambda v: v == np.nan,
        value=0,
        skip=(False, False, False),
    ):

        mesh = field.region.mesh
        dof = field.indices.dof

        self.ndim = field.dim  # mesh.ndim
        self.name = name
        self.value = value
        self.skip = np.array(skip).astype(int)[: mesh.ndim]  # self.ndim
        self.fun = [fx, fy, fz][: mesh.ndim]

        # apply functions on the points per coordinate
        # fx(x), fy(y), fz(z) and create a mask for each coordinate
        mask = [f(x) for f, x in zip(self.fun, mesh.points.T)]

        # combine the masks with "logical_or" if ndim > 1
        if mesh.ndim == 1:
            mask = mask[0]

        elif mesh.ndim == 2:
            mask = np.logical_or(mask[0], mask[1])

        elif mesh.ndim == 3:  # and mesh.points.shape[1] == 3:
            tmp = np.logical_or(mask[0], mask[1])
            mask = np.logical_or(tmp, mask[2])

        # tile the mask
        self.mask = np.tile(mask.reshape(-1, 1), self.ndim)

        # check if some axes should be skipped
        if True not in skip:
            pass
        else:
            # exclude mask from axes which should be skipped
            self.mask[:, np.where(self.skip)[0]] = False

        self.dof = dof[self.mask]
        self.points = np.arange(mesh.npoints)[mask]


def uniaxial(field, right=1, move=0.2, clamped=True):
    "Define boundaries for uniaxial loading."

    f1 = lambda x: np.isclose(x, right)

    bounds = symmetry(field)

    if clamped:
        bounds["right"] = Boundary(field, fx=f1, skip=(1, 0, 0))

    bounds["move"] = Boundary(field, fx=f1, skip=(0, 1, 1), value=move)

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dof0, dof1, ext0

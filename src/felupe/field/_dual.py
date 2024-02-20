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

from ..region._templates import (
    RegionBiQuadraticQuad,
    RegionConstantHexahedron,
    RegionConstantQuad,
    RegionHexahedron,
    RegionLagrange,
    RegionQuad,
    RegionQuadraticHexahedron,
    RegionQuadraticQuad,
    RegionQuadraticTetra,
    RegionQuadraticTriangle,
    RegionTetra,
    RegionTetraMINI,
    RegionTriangle,
    RegionTriangleMINI,
    RegionTriQuadraticHexahedron,
)
from ._base import Field


class FieldDual(Field):
    r"""A dual field on points of a :class:`~felupe.Region` with dimension ``dim`` and
    initial point ``values``.

    Parameters
    ----------
    region : Region
        The region on which the field will be created.
    dim : int, optional
        The dimension of the field  (default is 1).
    values : float or array
        A single value for all components of the field or an array of shape
        `(region.mesh.npoints, dim)`. Default is 0.0.
    offset : int, optional
        Offset for cell connectivity of the dual mesh (default is 0).
    npoints : int or None, optional
        Specified number of mesh points for the dual mesh (default is None).
    mesh: Mesh or None, optional
        A mesh which is used for the dual region (default is None). If None, the mesh
        is taken from the region.
    disconnect : bool or None, optional
        A flag to disconnect the dual mesh (default is None). If None, a disconnected
        mesh is used except for regions with quadratic-triangle or -tetra or MINI
        element formulations.
    **kwargs : dict, optional
        Optional keyword arguments for the dual region.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube(n=6)
    >>> region = fem.RegionHexahedron(mesh)
    >>>
    >>> displacement = fem.Field(region, dim=3)
    >>> pressure = fem.FieldDual(region)
    >>>
    >>> field = fem.FieldContainer([displacement, pressure])

    See Also
    --------
    felupe.FieldContainer : A container which holds one or multiple (mixed) fields.
    felupe.Field : Field on points of a :class:`~felupe.Region` with dimension ``dim``
        and initial point ``values``.
    felupe.FieldAxisymmetric : Axisymmetric field on points of a
        :class:`~felupe.Region` with dimension ``dim`` and initial point ``values``.
    felupe.FieldPlaneStrain : Plane strain field on points of a
        :class:`~felupe.Region` with dimension ``dim`` and initial point ``values``.
    felupe.mesh.dual : Create a dual :class:`~felupe.Mesh`.

    """

    def __init__(
        self,
        region,
        dim=1,
        values=0.0,
        offset=0,
        npoints=None,
        mesh=None,
        disconnect=None,
        **kwargs,
    ):
        # create dual regions
        region_dual_dict = {
            RegionHexahedron: RegionConstantHexahedron,
            RegionQuad: RegionConstantQuad,
            RegionQuadraticQuad: RegionConstantQuad,
            RegionBiQuadraticQuad: RegionQuad,
            RegionQuadraticHexahedron: RegionConstantHexahedron,
            RegionTriQuadraticHexahedron: RegionHexahedron,
            RegionQuadraticTetra: RegionTetra,
            RegionQuadraticTriangle: RegionTriangle,
            RegionTetraMINI: RegionTetra,
            RegionTriangleMINI: RegionTriangle,
            RegionLagrange: RegionLagrange,
        }

        mesh_kwargs = {
            RegionHexahedron: {},
            RegionQuad: {},
            RegionQuadraticQuad: {},
            RegionBiQuadraticQuad: {},
            RegionQuadraticHexahedron: {},
            RegionTriQuadraticHexahedron: {},
            RegionQuadraticTetra: {"disconnect": False},
            RegionQuadraticTriangle: {"disconnect": False},
            RegionTetraMINI: {"disconnect": False},
            RegionTriangleMINI: {"disconnect": False},
            RegionLagrange: {},
        }[type(region)]

        if disconnect is not None:
            mesh_kwargs["disconnect"] = disconnect

        points_per_cell = {
            RegionConstantHexahedron: 1,
            RegionConstantQuad: 1,
            RegionQuad: 4,
            RegionHexahedron: 8,
            RegionTriangle: 3,
            RegionTetra: 4,
            RegionLagrange: None,
        }

        kwargs0 = {"quadrature": region.quadrature, "grad": False}

        if isinstance(region, RegionLagrange):
            kwargs0["order"] = region.order - 1
            kwargs0["dim"] = region.mesh.dim
            points_per_cell[RegionLagrange] = region.order**region.element.dim

        RegionDual = region_dual_dict[type(region)]

        if mesh is None and points_per_cell[RegionDual] is not None:
            mesh = region.mesh.dual(
                points_per_cell=points_per_cell[RegionDual],
                offset=offset,
                npoints=npoints,
                **mesh_kwargs,
            )

        region_dual = RegionDual(mesh, **{**kwargs0, **kwargs})

        super().__init__(region_dual, dim=dim, values=values)

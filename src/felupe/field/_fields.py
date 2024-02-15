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
from ._axi import FieldAxisymmetric
from ._base import Field
from ._container import FieldContainer
from ._planestrain import FieldPlaneStrain


class FieldsMixed(FieldContainer):
    r"""A container with multiple (mixed) :class:`Fields <felupe.Field>` based on a
    :class:`Region`.

    Parameters
    ----------
    region : RegionHexahedron, RegionQuad, RegionQuadraticQuad, RegionBiQuadraticQuad, RegionQuadraticHexahedron, RegionTriQuadraticHexahedron, RegionQuadraticTetra, RegionQuadraticTriangle, RegionTetraMINI, RegionTriangleMINI or RegionLagrange
        A template region.
    n : int, optional
        Number of fields where the first one is a vector field of mesh-
        dimension and the following fields are scalar-fields (default is 3).
    values : tuple of float or tuple of ndarray, optional
        Initial field values (default is (0.0, 0.0, 1.0, 0.0)).
    axisymmetric : bool, optional
        Flag to initiate a :class:`FieldAxisymmetric` as the first field (default is
        False).
    planestrain : bool, optional
        Flag to initiate a :class:`FieldPlaneStrain` as the first field (default is
        False).
    offset : int, optional
        Offset for cell connectivity of the dual mesh (default is 0).
    npoints : int or None, optional
        Specified number of mesh points for the dual mesh (default is None).
    mesh: Mesh or None, optional
        A mesh which is used for the dual region (default is None). If None, the mesh
        is taken from the region.

    Notes
    -----
    The dual region is chosen automatically, i.e. for a :class:`RegionHexahedron` the
    dual region is :class:`RegionConstantHexahedron`. A total number of ``n`` fields
    are generated inside a :class:`FieldContainer`. For compatibility with
    :class:`~felupe.ThreeFieldVariation`, the third field is created with ones, all
    values of the other fields are initiated with zeros by default.

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
        n=3,
        values=(0.0, 0.0, 1.0, 0.0),
        axisymmetric=False,
        planestrain=False,
        offset=0,
        npoints=None,
        mesh=None,
        **kwargs,
    ):
        if axisymmetric is False and planestrain is False:
            F = Field
        elif axisymmetric is True and planestrain is False:
            F = FieldAxisymmetric
        elif axisymmetric is False and planestrain is True:
            F = FieldPlaneStrain

        fields = [F(region, dim=region.mesh.dim, values=values[0])]

        # create dual regions
        if n > 1:
            regions = {
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
            }
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

            RegionDual = regions[type(region)]

            if mesh is None and points_per_cell[RegionDual] is not None:
                mesh = region.mesh.dual(
                    points_per_cell=points_per_cell[RegionDual],
                    offset=offset,
                    npoints=npoints,
                    **mesh_kwargs[type(region)],
                )

            region_dual = RegionDual(
                mesh,
                **{**kwargs0, **kwargs},
            )

        for a in range(1, n):
            fields.append(Field(region_dual, values=values[a]))

        super().__init__(fields=tuple(fields))

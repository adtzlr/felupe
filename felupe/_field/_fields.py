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

from ..region._templates import (
    RegionBiQuadraticQuad,
    RegionConstantHexahedron,
    RegionConstantQuad,
    RegionHexahedron,
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
    r"""A mixed field based on a region and returns a :class:`FieldContainer`
    instance."""

    def __init__(
        self,
        region,
        n=3,
        values=(0, 0, 1, 0),
        axisymmetric=False,
        planestrain=False,
        offset=0,
        npoints=None,
    ):
        r"""Create a mixed field based on a region. The dual region is chosen
        automatically, i.e. for a :class:`RegionHexahedron` the dual region
        is :class:`RegionConstantHexahedron`. A total number of ``n`` fields
        are generated and passed to :class:`FieldContainer`.

        Arguments
        ---------
        region : Region
            A template region.
        n : int, optional (default is 2)
            Number of fields where the first one is a vector field of mesh-
            dimension and the following fields are scalar-fields.
        values : tuple, optional (default is (0, 0, 1, 0))
            Initial field values.
        axisymmetric : bool, optional
            Flag to initiate a axisymmetric Field (default is False).
        planestrain : bool, optional
            Flag to initiate a plane strain Field (default is False).
        offset : int, optional
            Offset for cell connectivity (default is 0).
        npoints : int, optional
            Specified number of mesh points (default is None).
        """

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
        }

        kwargs = {}
        if offset > 0:
            kwargs["offset"] = offset
        if npoints is not None:
            kwargs["npoints"] = npoints

        region_dual = regions[type(region)](
            region.mesh, quadrature=region.quadrature, grad=False, **kwargs
        )

        if axisymmetric is False and planestrain is False:
            F = Field
        elif axisymmetric is True and planestrain is False:
            F = FieldAxisymmetric
        elif axisymmetric is False and planestrain is True:
            F = FieldPlaneStrain

        fields = [F(region, dim=region.mesh.dim, values=values[0])]

        for a in range(1, n):
            fields.append(Field(region_dual, values=values[a]))

        super().__init__(fields=tuple(fields))

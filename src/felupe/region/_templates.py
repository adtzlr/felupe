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

from ..element import (
    ArbitraryOrderLagrange,
    BiQuadraticQuad,
    ConstantHexahedron,
    ConstantQuad,
    Hexahedron,
    Quad,
    QuadraticHexahedron,
    QuadraticQuad,
    QuadraticTetra,
    QuadraticTriangle,
    Tetra,
    TetraMINI,
    Triangle,
    TriangleMINI,
    TriQuadraticHexahedron,
    Vertex,
)
from ..mesh import Mesh
from ..quadrature import GaussLegendre, GaussLegendreBoundary
from ..quadrature import Tetrahedron as TetraQuadrature
from ..quadrature import Triangle as TriangleQuadrature
from ._boundary import RegionBoundary
from ._region import Region


class RegionConstantQuad(Region):
    "A region with a constant quad element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendre(order=1, dim=2),
        grad=False,
        **kwargs,
    ):
        element = ConstantQuad()
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionQuad(Region):
    r"""A region with a quad element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle()
       >>> region = fem.RegionQuad(mesh)
       >>> region
       <felupe Region object>
         Element formulation: Quad
         Quadrature rule: GaussLegendre
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=GaussLegendre(order=1, dim=2), grad=True, **kwargs
    ):
        if "container" in type(mesh).__name__.lower():
            raise TypeError(
                "A mesh container is not supported by a region, use a mesh instead."
            )

        element = Quad()

        if len(mesh.cells.T) > 4:
            mesh = Mesh(mesh.points, mesh.cells[:, :4], "quad")

        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionQuadraticQuad(Region):
    r"""A region with a (serendipity) quadratic quad element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle().add_midpoints_edges()
       >>> region = fem.RegionQuadraticQuad(mesh)
       >>> region
       <felupe Region object>
         Element formulation: QuadraticQuad
         Quadrature rule: GaussLegendre
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=GaussLegendre(order=2, dim=2), grad=True, **kwargs
    ):
        if "container" in type(mesh).__name__.lower():
            raise TypeError(
                "A mesh container is not supported by a region, use a mesh instead."
            )

        element = QuadraticQuad()

        if len(mesh.cells.T) > 8:
            mesh = Mesh(mesh.points, mesh.cells[:, :8], "quad8")

        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionBiQuadraticQuad(Region):
    r"""A region with a bi-quadratic (Lagrange) quad element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> rect = fem.Rectangle()
       >>> mesh = rect.add_midpoints_edges().add_midpoints_faces()
       >>> region = fem.RegionBiQuadraticQuad(mesh)
       >>> region
       <felupe Region object>
         Element formulation: BiQuadraticQuad
         Quadrature rule: GaussLegendre
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=GaussLegendre(order=2, dim=2), grad=True, **kwargs
    ):
        element = BiQuadraticQuad()

        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionQuadBoundary(RegionBoundary):
    """A region with a quad element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle()
       >>> region = fem.RegionQuadBoundary(mesh)
       >>> region
       <felupe Region object>
         Element formulation: Quad
         Quadrature rule: GaussLegendreBoundary
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()

    See Also
    --------
    felupe.RegionBoundary : A numeric boundary-region as a combination of a mesh, an
        element and a numeric integration scheme (quadrature rule).
    """

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=1, dim=2),
        grad=True,
        only_surface=True,
        mask=None,
        ensure_3d=False,
        **kwargs,
    ):
        element = Quad()

        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            ensure_3d=ensure_3d,
            **kwargs,
        )


class RegionQuadraticQuadBoundary(RegionBoundary):
    r"""A boundary region with a quadratic quad element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle().add_midpoints_edges()
       >>> region = fem.RegionQuadraticQuadBoundary(mesh)
       >>> region
       <felupe Region object>
         Element formulation: QuadraticQuad
         Quadrature rule: GaussLegendreBoundary
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()

    See Also
    --------
    felupe.RegionBoundary : A numeric boundary-region as a combination of a mesh, an
        element and a numeric integration scheme (quadrature rule).
    """

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=2),
        grad=True,
        only_surface=True,
        mask=None,
        ensure_3d=False,
        **kwargs,
    ):
        element = QuadraticQuad()

        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            ensure_3d=ensure_3d,
            **kwargs,
        )


class RegionBiQuadraticQuadBoundary(RegionBoundary):
    """A boundary region with a bi-quadratic quad element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle().add_midpoints_edges().add_midpoints_faces()
       >>> region = fem.RegionBiQuadraticQuadBoundary(mesh)
       >>> region
       <felupe Region object>
         Element formulation: BiQuadraticQuad
         Quadrature rule: GaussLegendreBoundary
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()

    See Also
    --------
    felupe.RegionBoundary : A numeric boundary-region as a combination of a mesh, an
        element and a numeric integration scheme (quadrature rule).
    """

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=2),
        grad=True,
        only_surface=True,
        mask=None,
        ensure_3d=False,
        **kwargs,
    ):
        element = BiQuadraticQuad()

        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            ensure_3d=ensure_3d,
            **kwargs,
        )


class RegionConstantHexahedron(Region):
    "A region with a constant hexahedron element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendre(order=1, dim=3),
        grad=False,
        **kwargs,
    ):
        element = ConstantHexahedron()
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionHexahedron(Region):
    r"""A region with a hexahedron element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube()
       >>> region = fem.RegionHexahedron(mesh)
       >>> region
       <felupe Region object>
         Element formulation: Hexahedron
         Quadrature rule: GaussLegendre
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=GaussLegendre(order=1, dim=3), grad=True, **kwargs
    ):
        if "container" in type(mesh).__name__.lower():
            raise TypeError(
                "A mesh container is not supported by a region, use a mesh instead."
            )

        element = Hexahedron()

        if len(mesh.cells.T) > 8:
            mesh = Mesh(mesh.points, mesh.cells[:, :8], "hexahedron")

        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionHexahedronBoundary(RegionBoundary):
    """A boundary region with a hexahedron element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube()
       >>> region = fem.RegionHexahedronBoundary(mesh)
       >>> region
       <felupe Region object>
         Element formulation: Hexahedron
         Quadrature rule: GaussLegendreBoundary
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()

    See Also
    --------
    felupe.RegionBoundary : A numeric boundary-region as a combination of a mesh, an
        element and a numeric integration scheme (quadrature rule).
    """

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=1, dim=3),
        grad=True,
        only_surface=True,
        mask=None,
        **kwargs,
    ):
        element = Hexahedron()
        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            **kwargs,
        )


class RegionQuadraticHexahedron(Region):
    """A region with a (serendipity) quadratic hexahedron element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube().add_midpoints_edges()
       >>> region = fem.RegionQuadraticHexahedron(mesh)
       >>> region
       <felupe Region object>
         Element formulation: QuadraticHexahedron
         Quadrature rule: GaussLegendre
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=GaussLegendre(order=2, dim=3), grad=True, **kwargs
    ):
        if "container" in type(mesh).__name__.lower():
            raise TypeError(
                "A mesh container is not supported by a region, use a mesh instead."
            )

        element = QuadraticHexahedron()

        if len(mesh.cells.T) > 20:
            mesh = Mesh(mesh.points, mesh.cells[:, :20], "hexahedron20")

        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionQuadraticHexahedronBoundary(RegionBoundary):
    """A boundary region with a (serendipity) quadratic hexahedron element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube().add_midpoints_edges()
       >>> region = fem.RegionQuadraticHexahedronBoundary(mesh)
       >>> region
       <felupe Region object>
         Element formulation: QuadraticHexahedron
         Quadrature rule: GaussLegendreBoundary
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()

    See Also
    --------
    felupe.RegionBoundary : A numeric boundary-region as a combination of a mesh, an
        element and a numeric integration scheme (quadrature rule).
    """

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=3),
        grad=True,
        only_surface=True,
        mask=None,
        **kwargs,
    ):
        element = QuadraticHexahedron()
        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            **kwargs,
        )


class RegionTriQuadraticHexahedron(Region):
    """A region with a tri-quadratic (Lagrange) hexahedron element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> cube = fem.Cube().add_midpoints_edges()
       >>> mesh = cube.add_midpoints_faces().add_midpoints_volumes()
       >>> region = fem.RegionTriQuadraticHexahedron(mesh)
       >>> region
       <felupe Region object>
         Element formulation: TriQuadraticHexahedron
         Quadrature rule: GaussLegendre
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=GaussLegendre(order=2, dim=3), grad=True, **kwargs
    ):
        element = TriQuadraticHexahedron()
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionTriQuadraticHexahedronBoundary(RegionBoundary):
    """A boundary region with a tri-quadratic (Lagrange) hexahedron element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> cube = fem.Cube().add_midpoints_edges()
       >>> mesh = cube.add_midpoints_faces().add_midpoints_volumes()
       >>> region = fem.RegionTriQuadraticHexahedronBoundary(mesh)
       >>> region
       <felupe Region object>
         Element formulation: TriQuadraticHexahedron
         Quadrature rule: GaussLegendreBoundary
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()

    See Also
    --------
    felupe.RegionBoundary : A numeric boundary-region as a combination of a mesh, an
        element and a numeric integration scheme (quadrature rule).
    """

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=3),
        grad=True,
        only_surface=True,
        mask=None,
        **kwargs,
    ):
        element = TriQuadraticHexahedron()
        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            **kwargs,
        )


class RegionLagrange(Region):
    """A region with an arbitrary order Lagrange element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.mesh.CubeArbitraryOrderHexahedron(order=3)
       >>> region = fem.RegionLagrange(mesh, order=3, dim=3)
       >>> region
       <felupe Region object>
         Element formulation: ArbitraryOrderLagrange
         Quadrature rule: GaussLegendre
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, order, dim, quadrature=None, grad=True, permute=True, **kwargs
    ):
        if quadrature is None:
            quadrature = GaussLegendre(order=order, dim=dim, permute=permute)

        element = ArbitraryOrderLagrange(order, dim, permute=permute)
        self.order = order

        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionTriangle(Region):
    """A region with a triangle element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.mesh.Rectangle().triangulate()
       >>> region = fem.RegionTriangle(mesh)
       >>> region
       <felupe Region object>
         Element formulation: Triangle
         Quadrature rule: Triangle
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=TriangleQuadrature(order=1), grad=True, **kwargs
    ):
        if "container" in type(mesh).__name__.lower():
            raise TypeError(
                "A mesh container is not supported by a region, use a mesh instead."
            )

        element = Triangle()

        if len(mesh.cells.T) > 3:
            m = Mesh(mesh.points, mesh.cells[:, :3], "triangle")
        else:
            m = mesh

        super().__init__(m, element, quadrature, grad=grad, **kwargs)


class RegionTetra(Region):
    """A region with a tetra element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube().triangulate()
       >>> region = fem.RegionTetra(mesh)
       >>> region
       <felupe Region object>
         Element formulation: Tetra
         Quadrature rule: Tetrahedron
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(self, mesh, quadrature=TetraQuadrature(order=1), grad=True, **kwargs):
        if "container" in type(mesh).__name__.lower():
            raise TypeError(
                "A mesh container is not supported by a region, use a mesh instead."
            )

        element = Tetra()

        if len(mesh.cells.T) > 4:
            m = Mesh(mesh.points, mesh.cells[:, :4], "tetra")
        else:
            m = mesh

        super().__init__(m, element, quadrature, grad=grad, **kwargs)


class RegionTriangleMINI(Region):
    """A region with a triangle-MINI element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle().triangulate().add_midpoints_faces()
       >>> region = fem.RegionTriangleMINI(mesh)
       >>> region
       <felupe Region object>
         Element formulation: TriangleMINI
         Quadrature rule: Triangle
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self,
        mesh,
        quadrature=TriangleQuadrature(order=2),
        grad=True,
        bubble_multiplier=0.1,
        **kwargs,
    ):
        element = TriangleMINI(bubble_multiplier=bubble_multiplier)
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionTetraMINI(Region):
    """A region with a tetra-MINI element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube().triangulate().add_midpoints_volumes()
       >>> region = fem.RegionTetraMINI(mesh)
       >>> region
       <felupe Region object>
         Element formulation: TetraMINI
         Quadrature rule: Tetrahedron
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self,
        mesh,
        quadrature=TetraQuadrature(order=2),
        grad=True,
        bubble_multiplier=0.1,
        **kwargs,
    ):
        element = TetraMINI(bubble_multiplier=bubble_multiplier)
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionQuadraticTriangle(Region):
    """A region with a quadratic triangle element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Rectangle().triangulate().add_midpoints_edges()
       >>> region = fem.RegionQuadraticTriangle(mesh)
       >>> region
       <felupe Region object>
         Element formulation: QuadraticTriangle
         Quadrature rule: Triangle
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(
        self, mesh, quadrature=TriangleQuadrature(order=2), grad=True, **kwargs
    ):
        element = QuadraticTriangle()
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionQuadraticTetra(Region):
    """A region with a quadratic tetra element.

    Examples
    --------
    Plot the element with its point-ids and the applied quadrature rule.

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> mesh = fem.Cube().triangulate().add_midpoints_edges()
       >>> region = fem.RegionQuadraticTetra(mesh)
       >>> region
       <felupe Region object>
         Element formulation: QuadraticTetra
         Quadrature rule: Tetrahedron
         Gradient evaluated: True
         Hessian evaluated: False

       >>> region.plot().show()
    """

    def __init__(self, mesh, quadrature=TetraQuadrature(order=2), grad=True, **kwargs):
        element = QuadraticTetra()
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)


class RegionVertex(Region):
    "A region with a vertex element."

    def __init__(
        self, mesh, quadrature=GaussLegendre(order=0, dim=1), grad=False, **kwargs
    ):
        element = Vertex()
        super().__init__(mesh, element, quadrature, grad=grad, **kwargs)

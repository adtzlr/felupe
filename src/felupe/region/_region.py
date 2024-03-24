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
from copy import deepcopy

import numpy as np

from ..math import det, inv


class Region:
    r"""
    A numeric region as a combination of a mesh, an element and a numeric integration
    scheme (quadrature rule).

    Parameters
    ----------
    mesh : Mesh
        A mesh with points and cells.
    element : Element
        The finite element formulation to be applied on the cells.
    quadrature: Quadrature
        An element-compatible numeric integration scheme with points and weights.
    grad : bool, optional
        A flag to invoke gradient evaluation (default is True). If True, the partial
        derivatives of the element shape functions w.r.t. undeformed coordinates
        :math:`\frac{\partial \boldsymbol{h}(\boldsymbol{r})}{\partial \boldsymbol{X}}`
        and the differential volumes :math:`dV` are evaluated.

    Attributes
    ----------
    mesh : Mesh
        A mesh with points and cells.
    element : Finite element
        The finite element formulation to be applied on the cells.
    quadrature: Quadrature scheme
        An element-compatible numeric integration scheme with points and weights.
    h : ndarray
        Element shape function array ``h_aq`` of shape function ``a`` evaluated at
        quadrature point ``q``.
    dhdr : ndarray
        Partial derivative of element shape function array ``dhdr_aJq`` with shape
        function ``a`` w.r.t. natural element coordinate ``J`` evaluated at quadrature
        point ``q`` for every cell ``c`` (geometric gradient or **Jacobian**
        transformation between ``X`` and ``r``).
    dXdr : ndarray
        Geometric gradient ``dXdr_IJqc`` as partial derivative of undeformed coordinate
        ``I`` w.r.t. natural element coordinate ``J`` evaluated at quadrature point
        ``q`` for every cell ``c`` (geometric gradient or **Jacobian** transformation
        between ``X`` and ``r``).
    drdX : ndarray
        Inverse of dXdr.
    dV : ndarray
        Numeric *differential volume element* as product of determinant of geometric
        gradient  ``dV_qc = det(dXdr)_qc w_q`` and quadrature weight ``w_q``, evaluated
        at quadrature point ``q`` for every cell ``c``.
    dhdX : ndarray
        Partial derivative of element shape functions ``dhdX_aJqc`` of shape function
        ``a`` w.r.t. undeformed coordinate ``J`` evaluated at quadrature point ``q`` for
        every cell ``c``.

    Notes
    -----
    The gradients of the element shape functions w.r.t the undeformed coordinates are
    evaluated at all integration points of each cell in the region if the optional
    gradient argument is ``True``.

    .. math::

       \frac{\partial X_I}{\partial r_J} &= \hat{X}_{aI} \frac{
               \partial h_a}{\partial r_J
           }

       \frac{\partial h_a}{\partial X_J} &= \frac{\partial h_a}{\partial r_I}
       \frac{\partial r_I}{\partial X_J}

       dV &= \det\left(\frac{\partial X_I}{\partial r_J}\right) w

    Examples
    --------
    >>> import felupe as fem

    >>> mesh = fem.Rectangle(n=(3, 2))
    >>> element = fem.Quad()
    >>> quadrature = fem.GaussLegendre(order=1, dim=2)

    >>> region = fem.Region(mesh, element, quadrature)
    >>> region
    <felupe Region object>
      Element formulation: Quad
      Quadrature rule: GaussLegendre
      Gradient evaluated: True

    The numeric differential volumes are the products of the determinant of the
    geometric gradient :math:`\frac{\partial X_I}{\partial r_J}` and the weights `w` of
    the quadrature points. The differential volume array is of shape
    ``(nquadraturepoints, ncells)``.

    >>> region.dV
    array([[0.125, 0.125],
           [0.125, 0.125],
           [0.125, 0.125],
           [0.125, 0.125]])

    Cell-volumes may be obtained by cell-wise sums of the differential volumes located
    at the quadrature points.

    >>> region.dV.sum(axis=0)
    array([0.5, 0.5])

    The partial derivative of the first element shape function w.r.t. the undeformed
    coordinates evaluated at the second integration point of the last element of the
    region:

    >>> region.dhdX[0, :, 1, -1]
    array([-1.57735027, -0.21132487])

    """

    def __init__(self, mesh, element, quadrature, grad=True):
        self.evaluate_gradient = grad
        self.reload(mesh=mesh, element=element, quadrature=quadrature)

    def copy(self, mesh=None, element=None, quadrature=None):
        """Return a copy of the region and reload it if necessary.

        Parameters
        ----------
        mesh : Mesh or None, optional
            A mesh with points and cells (default is None).
        element : Element or None, optional
            The finite element formulation to be applied on the cells (default is None).
        quadrature: Quadrature or None, optional
            An element-compatible numeric integration scheme with points and weights
            (default is None).

        Returns
        -------
        Region
            A copy of the reloaded region.

        See Also
        --------
        felupe.Region.reload : Reload the numeric region inplace.
        """

        region = deepcopy(self)
        region.reload(mesh=mesh, element=element, quadrature=quadrature)

        return region

    def reload(self, mesh=None, element=None, quadrature=None):
        """Reload the numeric region inplace.

        Parameters
        ----------
        mesh : Mesh or None, optional
            A mesh with points and cells (default is None).
        element : Element or None, optional
            The finite element formulation to be applied on the cells (default is None).
        quadrature: Quadrature or None, optional
            An element-compatible numeric integration scheme with points and weights
            (default is None).

        Examples
        --------
        ..  warning::
            If the points of a mesh are modified and a region was already created with
            the mesh, it is important to re-evaluate (reload) the
            :class:`~felupe.Region` inplace.

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> new_points = mesh.rotate(angle_deg=-90, axis=2).points
        >>> mesh.update(points=new_points, callback=region.reload)

        See Also
        --------
        felupe.Mesh.update : Update the mesh with given points and cells arrays inplace.
            Optionally, a callback is evaluated.
        """

        region = self

        if mesh is not None:
            region.mesh = mesh

        if element is not None:
            region.element = element

        if quadrature is not None:
            region.quadrature = quadrature

        if mesh is not None or element is not None or quadrature is not None:
            # element shape function
            region.element.h = np.array(
                [region.element.function(q) for q in region.quadrature.points]
            ).T
            region.h = np.tile(np.expand_dims(region.element.h, -1), region.mesh.ncells)

            # partial derivative of element shape function
            region.element.dhdr = np.array(
                [region.element.gradient(q) for q in region.quadrature.points]
            ).transpose(1, 2, 0)
            region.dhdr = np.tile(
                np.expand_dims(region.element.dhdr, -1), region.mesh.ncells
            )

            if region.evaluate_gradient:
                # geometric gradient
                region.dXdr = np.einsum(
                    "caI,aJqc->IJqc", region.mesh.points[region.mesh.cells], region.dhdr
                )

                # determinant and inverse of dXdr
                J = det(region.dXdr)
                region.drdX = inv(region.dXdr, determinant=J)

                # numeric **differential volume element**
                region.dV = np.multiply(
                    J, region.quadrature.weights.reshape(-1, 1), out=J
                )

                # check for negative **differential volume elements**
                if np.any(region.dV < 0):
                    cells_negative_volume = np.where(np.any(region.dV < 0, axis=0))[0]
                    message_negative_volumes = "".join(
                        [
                            f"Negative volumes for cells \n {cells_negative_volume}\n",
                            "Try ``mesh.flip(np.any(region.dV < 0, axis=0))`` ",
                            "and re-create the region.",
                        ]
                    )
                    warnings.warn(message_negative_volumes)

                # Partial derivative of element shape function
                # w.r.t. undeformed coordinates
                region.dhdX = np.einsum("aIqc,IJqc->aJqc", region.dhdr, region.drdX)

    def __repr__(self):
        header = "<felupe Region object>"
        element = f"  Element formulation: {type(self.element).__name__}"
        quadrature = f"  Quadrature rule: {type(self.quadrature).__name__}"
        grad = f"  Gradient evaluated: {hasattr(self, 'dV')}"

        return "\n".join([header, element, quadrature, grad])

    def plot(self, **kwargs):
        """Plot the element with point-ids and the quadrature points,
        scaled by their weights."""

        return self.quadrature.plot(plotter=self.element.plot(**kwargs))

    def screenshot(
        self,
        filename=None,
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the element with the applied quadrature.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        if filename is None:
            filename = f"region-{self.element.cell_type}.png"

        return self.plot(off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

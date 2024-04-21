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

from copy import deepcopy as copy
import numpy as np

from ._view import ViewMaterial, ViewMaterialIncompressible


class ConstitutiveMaterial:
    "Base class for constitutive materials."

    def view(self, incompressible=False, **kwargs):
        """Create views on normal force per undeformed area vs. stretch curves for the
        elementary homogeneous deformations uniaxial tension/compression, planar shear
        and biaxial tension of a given isotropic material formulation.

        Parameters
        ----------
        incompressible : bool, optional
            A flag to enforce views on incompressible deformations (default is False).
        **kwargs : dict, optional
            Optional keyword-arguments for :class:`~felupe.ViewMaterial` or
            :class:`~felupe.ViewMaterialIncompressible`.

        Returns
        -------
        felupe.ViewMaterial or felupe.ViewMaterialIncompressible

        See Also
        --------
        felupe.ViewMaterial : Create views on normal force per undeformed area vs.
            stretch curves for the elementary homogeneous deformations uniaxial
            tension/compression, planar shear and biaxial tension of a given isotropic
            material formulation.
        felupe.ViewMaterialIncompressible : Create views on normal force per undeformed
            area vs. stretch curves for the elementary homogeneous incompressible
            deformations uniaxial tension/compression, planar shear and biaxial tension
            of a given isotropic material formulation.
        """

        View = ViewMaterial
        if incompressible:
            View = ViewMaterialIncompressible

        return View(self, **kwargs)

    def plot(self, incompressible=False, **kwargs):
        """Return a plot with normal force per undeformed area vs. stretch curves for
        the elementary homogeneous deformations uniaxial tension/compression, planar
        shear and biaxial tension of a given isotropic material formulation.

        Parameters
        ----------
        incompressible : bool, optional
            A flag to enforce views on incompressible deformations (default is False).
        **kwargs : dict, optional
            Optional keyword-arguments for :class:`~felupe.ViewMaterial` or
            :class:`~felupe.ViewMaterialIncompressible`.

        Returns
        -------
        matplotlib.axes.Axes

        See Also
        --------
        felupe.ViewMaterial : Create views on normal force per undeformed area vs.
            stretch curves for the elementary homogeneous deformations uniaxial
            tension/compression, planar shear and biaxial tension of a given isotropic
            material formulation.
        felupe.ViewMaterialIncompressible : Create views on normal force per undeformed
            area vs. stretch curves for the elementary homogeneous incompressible
            deformations uniaxial tension/compression, planar shear and biaxial tension
            of a given isotropic material formulation.
        """

        return self.view(incompressible=incompressible, **kwargs).plot()

    def screenshot(self, filename="umat.png", incompressible=False, **kwargs):
        """Save a screenshot with normal force per undeformed area vs. stretch curves
        for the elementary homogeneous deformations uniaxial tension/compression, planar
        shear and biaxial tension of a given isotropic material formulation.

        Parameters
        ----------
        filename : str, optional
            The filename of the screenshot (default is "umat.png").
        incompressible : bool, optional
            A flag to enforce views on incompressible deformations (default is False).
        **kwargs : dict, optional
            Optional keyword-arguments for :class:`~felupe.ViewMaterial` or
            :class:`~felupe.ViewMaterialIncompressible`.

        Returns
        -------
        matplotlib.axes.Axes

        See Also
        --------
        felupe.ViewMaterial : Create views on normal force per undeformed area vs.
            stretch curves for the elementary homogeneous deformations uniaxial
            tension/compression, planar shear and biaxial tension of a given isotropic
            material formulation.
        felupe.ViewMaterialIncompressible : Create views on normal force per undeformed
            area vs. stretch curves for the elementary homogeneous incompressible
            deformations uniaxial tension/compression, planar shear and biaxial tension
            of a given isotropic material formulation.
        """

        import matplotlib.pyplot as plt

        ax = self.plot(incompressible=incompressible, **kwargs)
        fig = ax.get_figure()
        fig.savefig(filename)
        plt.close(fig)

        return ax

    def optimize(self, ux=None, ps=None, bx=None, incompressible=False, **kwargs):
        """Optimize the material parameters by a least-squares fit on experimental
        stretch-stress data.

        Parameters
        ----------
        ux : array of shape (2, n) or None, optional
            Experimental uniaxial stretch and force-per-undeformed-area data (default is
            None).
        ps : array of shape (2, n) or None, optional
            Experimental planar-shear stretch and force-per-undeformed-area data
            (default is None).
        bx : array of shape (2, n) or None, optional
            Experimental biaxial stretch and force-per-undeformed-area data (default is
            None).
        incompressible : bool, optional
            A flag to enforce incompressible deformations (default is False).
        **kwargs : dict, optional
            Optional keyword arguments are passed to
            :func:`scipy.optimize.least_squares`.

        Returns
        -------
        ConstitutiveMaterial
            A copy of the constitutive material with the optimized material parameters.
        scipy.optimize.OptimizeResult
            Represents the optimization result.
        """
        from scipy.optimize import least_squares

        experiment = [(None, None) if lc is None else lc for lc in [ux, bx, ps]]

        def fun(values):
            for key, value in zip(self.kwargs.keys(), values):
                self.kwargs[key] = value
            model = self.view(
                incompressible=incompressible,
                ux=experiment[0][0],
                bx=experiment[1][0],
                ps=experiment[2][0],
            ).evaluate()
            residuals = [
                predicted[1] - observed[1]
                for predicted, observed in zip(model, experiment)
                if observed[1] is not None
            ]

            return np.concatenate(residuals)

        res = least_squares(fun=fun, x0=list(self.kwargs.values()), **kwargs)

        out = copy(self)
        for key, value in zip(self.kwargs.keys(), res.x):
            out.kwargs[key] = value

        return out, res

    def __and__(self, other_material):
        return CompositeMaterial(self, other_material)


class CompositeMaterial(ConstitutiveMaterial):
    """A composite material with two constitutive materials merged. State variables are
    only considered for the first material.

    Parameters
    ----------
    material : ConstitutiveMaterial
        First constitutive material.
    other_material : ConstitutiveMaterial
        Second constitutive material.

    Notes
    -----
    ..  warning::
        Do not merge two constitutive materials with the same keys of material
        parameters. In this case, the values of these material parameters are taken from
        the first constitutive material.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> nh = fem.NeoHooke(mu=1.0)
        >>> vol = fem.Volumetric(bulk=2.0)
        >>> umat = nh & vol
        >>> ax = umat.plot()

    """

    def __init__(self, material, other_material):
        self.materials = [material, other_material]
        self.kwargs = {**other_material.kwargs, **material.kwargs}
        self.x = material.x

    def gradient(self, x, **kwargs):
        gradients = [material.gradient(x, **kwargs) for material in self.materials]
        nfields = len(x) - 1
        P = [np.sum([grad[i] for grad in gradients], axis=0) for i in range(nfields)]
        statevars_new = gradients[0][-1]
        return [*P, statevars_new]

    def hessian(self, x, **kwargs):
        hessians = [material.hessian(x, **kwargs) for material in self.materials]
        nfields = len(x) - 1
        return [np.sum([hess[i] for hess in hessians], axis=0) for i in range(nfields)]

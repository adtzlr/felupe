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

import numpy as np


class ViewMaterialIncompressible:
    """View on normal force per undeformed area vs. stretch curves for the elementary
    homogeneous incompressible deformations uniaxial tension/compression, planar
    shear and biaxial tension of a given isotropic material formulation.

    Parameters
    ----------
    umat : class
        A class with methods for the gradient and hessian of the strain energy density
        function w.r.t. the deformation gradient. See :class:`~felupe.Material` for
        further details.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.extended_tube, Gc=0.2, Ge=0.2, beta=0.2, delta=0.1)
    >>> preview = ViewMaterialIncompressible(umat)
    >>> ax = preview.plot(show_title=True, show_kwargs=True)

    .. image:: images/preview_hyperelastic.png
       :width: 400px

    """

    def __init__(self, umat):
        self.umat = umat

    def uniaxial(self):
        """Normal force per undeformed area vs. stretch curve for a uniaxial
        incompressible deformation."""

        λ1 = np.linspace(0.7, 2.5)
        λ2 = λ3 = 1 / np.sqrt(λ1)
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, statevars = self.umat.gradient([F, None])
        return λ1, (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel()

    def planar(self):
        """Normal force per undeformed area vs. stretch curve for a planar shear
        incompressible deformation."""

        λ1 = np.linspace(1, 2.5)
        λ2 = np.ones_like(λ1)
        λ3 = 1 / λ1
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, statevars = self.umat.gradient([F, None])
        return λ1, (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel()

    def biaxial(self):
        """Normal force per undeformed area vs. stretch curve for a equi-biaxial
        incompressible deformation."""

        λ1 = λ2 = np.linspace(1, 1.75)
        λ3 = 1 / λ1**2
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, statevars = self.umat.gradient([F, None])
        return λ1, (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel()

    def evaluate(self):
        """Evaluate normal force per undeformed area vs. stretch curves for the
        elementary homogeneous incompressible deformations uniaxial tension/compression,
        planar shear and biaxial tension."""

        return [
            ("Uniaxial Tension/Compression", self.uniaxial()),
            ("Planar Shear", self.planar()),
            ("Biaxial Tension", self.biaxial()),
        ]

    def plot(self, ax=None, show_title=True, show_kwargs=True, **kwargs):
        """Plot normal force per undeformed area vs. stretch curves for the elementary
        homogeneous incompressible deformations uniaxial tension/compression, planar
        shear and biaxial tension."""

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        data = self.evaluate()
        for label, loadcase in data:
            ax.plot(*loadcase, label=label)

        ax.set_xlabel(r"Stretch $l/L$ $\rightarrow$")
        ax.set_ylabel("Normal force per \n undeformed area" + r" $N/A$ $\rightarrow$")
        ax.legend()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.plot(xlim, [0, 0], "black")
        ax.plot([1, 1], ylim, "black")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if show_title:
            title = self.umat.__class__.__name__
            if hasattr(self.umat, "fun"):
                title += (
                    " ("
                    + " ".join(
                        [name.title() for name in self.umat.fun.__name__.split("_")]
                    )
                    + ")"
                )
            fig.suptitle(title)

        if show_kwargs:
            if hasattr(self.umat, "kwargs"):
                ax.set_title(
                    ", ".join(
                        [f"{key}={value}" for key, value in self.umat.kwargs.items()]
                    ),
                    fontdict=dict(fontsize="small"),
                )
        return ax

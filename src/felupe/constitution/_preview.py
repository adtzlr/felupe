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

from ..math._math import linsteps


class PlotMaterial:
    "Plot force-stretch curves of constitutive material formulations."

    def evaluate(self):
        """Evaluate normal force per undeformed area vs. stretch curves for the
        elementary homogeneous incompressible deformations uniaxial tension/compression,
        planar shear and biaxial tension. A load case is not included if its array of
        stretches  (attribute ``ux``, ``ps`` or ``bx``) is None.

        Returns
        -------
        list of 3-tuple
            List with 3-tuple of stretch and force arrays and the label string for each
            load case.
        """

        data = []

        if self.ux is not None:
            data.append(self.uniaxial())

        if self.ps is not None:
            data.append(self.planar())

        if self.bx is not None:
            data.append(self.biaxial())

        return data

    def plot(self, ax=None, show_title=True, show_kwargs=True, **kwargs):
        """Plot normal force per undeformed area vs. stretch curves for the elementary
        homogeneous incompressible deformations uniaxial tension/compression, planar
        shear and biaxial tension."""

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        data = self.evaluate()
        for stretch, force, label in data:
            ax.plot(stretch, force, label=label)

        ax.set_xlabel(r"Stretch $l/L$ $\rightarrow$")
        ax.set_ylabel("Normal force per undeformed area" + r" $N/A$ $\rightarrow$")
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
                fun = self.umat.fun
                label = [fun.__class__.__name__]
                if callable(fun):
                    label = [name.title() for name in fun.__name__.split("_")]
                title += " (" + " ".join(label) + ")"
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


class ViewMaterial(PlotMaterial):
    """Create views on normal force per undeformed area vs. stretch curves for the
    elementary homogeneous deformations uniaxial tension/compression, planar shear and
    biaxial tension of a given isotropic material formulation.

    Parameters
    ----------
    umat : class
        A class with methods for the gradient and hessian of the strain energy density
        function w.r.t. the deformation gradient. See :class:`~felupe.Material` for
        further details.
    ux : ndarray, optional
        Array with stretches for uniaxial tension/compression. Default is
        ``linsteps([0.7, 2.5], num=36)``.
    ps : ndarray, optional
        Array with stretches for planar shear. Default is
        ``linsteps([1.0, 2.5], num=30)``.
    bx : ndarray, optional
        Array with stretches for equi-biaxial tension. Default is
        ``linsteps([1.0, 1.75], num=15)```.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> umat = fem.OgdenRoxburgh(fem.NeoHooke(mu=1, bulk=2), r=3, m=1, beta=0)
    >>> view = fem.ViewMaterial(
    >>>     umat,
    >>>     ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
    >>>     ps=None,
    >>>     bx=None,
    >>> )
    >>> ax = view.plot(show_title=True, show_kwargs=True)

    .. image:: images/umat.png
       :width: 400px

    """

    def __init__(
        self,
        umat,
        ux=linsteps([0.7, 2.5], num=36),
        ps=linsteps([1, 2.5], num=30),
        bx=linsteps([1, 1.75], num=15),
    ):
        self.umat = umat
        self.ux = ux
        self.ps = ps
        self.bx = bx
        self.statevars_included = self.umat.x[-1].size > 0

    def uniaxial(self, stretches=None):
        """Normal force per undeformed area vs. stretch curve for a uniaxial
        deformation.

        Parameters
        ----------
        stretches : ndarray or None, optional
            Array with stretches at which the forces are evaluated (default is None). If
            None, the stretches from initialization are used.

        Returns
        -------
        tuple of ndarray
            3-tuple with array of stretches and array of forces and the label.
        """

        if stretches is None:
            stretches = self.ux

        λ1 = stretches
        λ2 = λ3 = 1 / np.sqrt(λ1)
        eye = np.eye(3).reshape(3, 3, 1, 1)

        def fun(λ3):
            λ2 = λ3
            F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)
            if self.statevars_included:
                statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
                P = np.zeros_like(F)
                for increment, defgrad in enumerate(F.T):
                    P[..., [increment]], statevars = self.umat.gradient(
                        [F[..., [increment]], statevars]
                    )
            else:
                P, statevars = self.umat.gradient([F, None])
            return P[2, 2].ravel()

        from scipy.optimize import root

        λ2 = λ3 = root(fun, λ3).x
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        if self.statevars_included:
            statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
            P = np.zeros_like(F)
            for increment, defgrad in enumerate(F.T):
                P[..., [increment]], statevars = self.umat.gradient(
                    [F[..., [increment]], statevars]
                )
        else:
            P, statevars = self.umat.gradient([F, None])

        return λ1, P[0, 0].ravel(), "Uniaxial"

    def planar(self, stretches=None):
        """Normal force per undeformed area vs. stretch curve for a planar shear
        incompressible deformation.

        Parameters
        ----------
        stretches : ndarray or None, optional
            Array with stretches at which the forces are evaluated (default is None). If
            None, the stretches from initialization are used.

        Returns
        -------
        tuple of ndarray
            3-tuple with array of stretches and array of forces and the label.
        """

        if stretches is None:
            stretches = self.ps

        λ1 = stretches
        λ2 = np.ones_like(λ1)
        λ3 = 1 / λ1
        eye = np.eye(3).reshape(3, 3, 1, 1)

        def fun(λ3):
            F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)
            if self.statevars_included:
                statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
                P = np.zeros_like(F)
                for increment, defgrad in enumerate(F.T):
                    P[..., [increment]], statevars = self.umat.gradient(
                        [F[..., [increment]], statevars]
                    )
            else:
                P, statevars = self.umat.gradient([F, None])
            return P[2, 2].ravel()

        from scipy.optimize import root

        λ3 = root(fun, λ3).x
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        if self.statevars_included:
            statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
            P = np.zeros_like(F)
            for increment, defgrad in enumerate(F.T):
                P[..., [increment]], statevars = self.umat.gradient(
                    [F[..., [increment]], statevars]
                )
        else:
            P, statevars = self.umat.gradient([F, None])

        return λ1, P[0, 0].ravel(), "Planar Shear"

    def biaxial(self, stretches=None):
        """Normal force per undeformed area vs. stretch curve for a equi-biaxial
        incompressible deformation.

        Parameters
        ----------
        stretches : ndarray or None, optional
            Array with stretches at which the forces are evaluated (default is None). If
            None, the stretches from initialization are used.

        Returns
        -------
        tuple of ndarray
            3-tuple with array of stretches and array of forces and the label.
        """

        if stretches is None:
            stretches = self.bx

        λ1 = λ2 = stretches
        λ3 = 1 / λ1**2
        eye = np.eye(3).reshape(3, 3, 1, 1)

        def fun(λ3):
            F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)
            if self.statevars_included:
                statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
                P = np.zeros_like(F)
                for increment, defgrad in enumerate(F.T):
                    P[..., [increment]], statevars = self.umat.gradient(
                        [F[..., [increment]], statevars]
                    )
            else:
                P, statevars = self.umat.gradient([F, None])
            return P[2, 2].ravel()

        from scipy.optimize import root

        λ3 = root(fun, λ3).x
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        if self.statevars_included:
            statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
            P = np.zeros_like(F)
            for increment, defgrad in enumerate(F.T):
                P[..., [increment]], statevars = self.umat.gradient(
                    [F[..., [increment]], statevars]
                )
        else:
            P, statevars = self.umat.gradient([F, None])

        return λ1, P[0, 0].ravel(), "Biaxial"


class ViewMaterialIncompressible(PlotMaterial):
    """Create views on normal force per undeformed area vs. stretch curves for the
    elementary homogeneous incompressible deformations uniaxial tension/compression,
    planar shear and biaxial tension of a given isotropic material formulation.

    Parameters
    ----------
    umat : class
        A class with methods for the gradient and hessian of the strain energy density
        function w.r.t. the deformation gradient. See :class:`~felupe.Material` for
        further details.
    ux : ndarray, optional
        Array with stretches for incompressible uniaxial tension/compression. Default is
        ``linsteps([0.7, 2.5], num=36)``.
    ps : ndarray, optional
        Array with stretches for incompressible planar shear. Default is
        ``linsteps([1, 2.5], num=30)``.
    bx : ndarray, optional
        Array with stretches for incompressible equi-biaxial tension. Default is
        ``linsteps([1, 1.75], num=15)``.

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> umat = fem.Hyperelastic(fem.extended_tube, Gc=0.2, Ge=0.2, beta=0.2, delta=0.1)
    >>> preview = fem.ViewMaterialIncompressible(umat)
    >>> ax = preview.plot(show_title=True, show_kwargs=True)

    .. image:: images/preview_hyperelastic.png
       :width: 400px

    >>> umat = fem.OgdenRoxburgh(fem.NeoHooke(mu=1), r=3, m=1, beta=0)
    >>> view = fem.ViewMaterialIncompressible(
    >>>     umat,
    >>>     ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
    >>>     ps=None,
    >>>     bx=None,
    >>> )
    >>> ax = view.plot(show_title=True, show_kwargs=True)

    .. image:: images/umat_incompressible.png
       :width: 400px

    """

    def __init__(
        self,
        umat,
        ux=linsteps([0.7, 2.5], num=36),
        ps=linsteps([1, 2.5], num=30),
        bx=linsteps([1, 1.75], num=15),
    ):
        self.umat = umat
        self.ux = ux
        self.ps = ps
        self.bx = bx
        self.statevars_included = self.umat.x[-1].size > 0

    def uniaxial(self, stretches=None):
        """Normal force per undeformed area vs. stretch curve for a uniaxial
        incompressible deformation.

        Parameters
        ----------
        stretches : ndarray or None, optional
            Array with stretches at which the forces are evaluated (default is None). If
            None, the stretches from initialization are used.

        Returns
        -------
        tuple of ndarray
            3-tuple with array of stretches and array of forces and the label.
        """

        if stretches is None:
            stretches = self.ux

        λ1 = stretches
        λ2 = λ3 = 1 / np.sqrt(λ1)
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        if self.statevars_included:
            statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
            P = np.zeros_like(F)
            for increment, defgrad in enumerate(F.T):
                P[..., [increment]], statevars = self.umat.gradient(
                    [F[..., [increment]], statevars]
                )
        else:
            P, statevars = self.umat.gradient([F, None])

        return λ1, (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel(), "Uniaxial (Incompressible)"

    def planar(self, stretches=None):
        """Normal force per undeformed area vs. stretch curve for a planar shear
        incompressible deformation.

        Parameters
        ----------
        stretches : ndarray or None, optional
            Array with stretches at which the forces are evaluated (default is None). If
            None, the stretches from initialization are used.

        Returns
        -------
        tuple of ndarray
            3-tuple with array of stretches and array of forces and the label.
        """

        if stretches is None:
            stretches = self.ps

        λ1 = stretches
        λ2 = np.ones_like(λ1)
        λ3 = 1 / λ1
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        if self.statevars_included:
            statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
            P = np.zeros_like(F)
            for increment, defgrad in enumerate(F.T):
                P[..., [increment]], statevars = self.umat.gradient(
                    [F[..., [increment]], statevars]
                )
        else:
            P, statevars = self.umat.gradient([F, None])

        return (
            λ1,
            (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel(),
            "Planar Shear (Incompressible)",
        )

    def biaxial(self, stretches=None):
        """Normal force per undeformed area vs. stretch curve for a equi-biaxial
        incompressible deformation.

        Parameters
        ----------
        stretches : ndarray or None, optional
            Array with stretches at which the forces are evaluated (default is None). If
            None, the stretches from initialization are used.

        Returns
        -------
        tuple of ndarray
            3-tuple with array of stretches and array of forces and the label.
        """

        if stretches is None:
            stretches = self.bx

        λ1 = λ2 = stretches
        λ3 = 1 / λ1**2
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        if self.statevars_included:
            statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))
            P = np.zeros_like(F)
            for increment, defgrad in enumerate(F.T):
                P[..., [increment]], statevars = self.umat.gradient(
                    [F[..., [increment]], statevars]
                )
        else:
            P, statevars = self.umat.gradient([F, None])

        return λ1, (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel(), "Biaxial (Incompressible)"


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
    >>> import felupe as fem
    >>>
    >>> nh = fem.NeoHooke(mu=1.0)
    >>> vol = fem.Volumetric(bulk=2.0)
    >>> umat = nh & vol
    >>> ax = umat.plot()

    ..  image:: images/umat_composite.png
        :width: 400px

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

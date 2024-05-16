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
    r"""Base class for constitutive materials.

    A constitutive material definition, or so-called ``umat`` (user material), is a
    class with methods for evaluating gradients and hessians of the strain energy
    density function with respect to the defined fields in the field container, where
    the gradient of the first (displacement) field is passed as the deformation
    gradient. For all following fields, the field values (no gradients) are provided. An
    attribute ``x=[np.zeros(statevars_shape)]`` has to be added to the class to define
    the shape of optional state variables. For reasons of performance, FElupe passes the
    field gradients and values *all at once*, e.g. the deformation gradient is of shape
    ``(3, 3, q, c)``, where ``q`` refers to the number of quadrature points per cell and
    ``c`` to the number of cells. These last two axes are the so-called *trailing axes*.
    Math-functions from :ref:`felupe.math <felupe-api-math>` all support the operation
    on trailing axes. The constitutive material definition class should be inherited
    from :class:`~felupe.ConstitutiveMaterial` in order to provide force-stretch curves
    for elementary deformations.

    Examples
    --------
    Take this code-block as a template for a two-field :math:`(\boldsymbol{u}, p)`
    formulation with the old displacement gradient as a state variable:

    ..  pyvista-plot::
        :context:

        import numpy as np
        import felupe as fem

        # math-functions which support trailing axes
        from felupe.math import det, dya, identity, transpose, inv

        class MyMaterialFormulation(fem.ConstitutiveMaterial):

            def __init__(self):
                # provide the shape of state variables without trailing axes
                # values are ignored - state variables are always initiated with zeros
                self.x = [np.zeros((3, 3))]

            def gradient(self, x):
                "Gradients of the strain energy density function."

                # extract variables
                F, p, statevars = x[0], x[1], x[-1]

                # user code
                dWdF = None  # first Piola-Kirchhoff stress tensor
                dWdp = None

                # update state variables
                # example: the displacement gradient
                statevars_new = F - identity(F)

                return [dWdF, dWdp, statevars_new]

            def hessian(self, x, **kwargs):
                "Hessians of the strain energy density function."

                # extract variables
                F, p, statevars = x[0], x[1], x[-1]

                # user code
                d2WdFdF = None  # fourth-order elasticity tensor
                d2WdFdp = None
                d2Wdpdp = None

                # upper-triangle items of the hessian
                return [d2WdFdF, d2WdFdp, d2Wdpdp]

        umat = MyMaterialFormulation()

    See Also
    --------
    felupe.constitutive_material : A decorator for a constitutive material definition.
    """

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

    def optimize(
        self, ux=None, ps=None, bx=None, incompressible=False, relative=False, **kwargs
    ):
        r"""Optimize the material parameters by a least-squares fit on experimental
        stretch-stress data.

        Parameters
        ----------
        ux : array of shape (2, ...) or None, optional
            Experimental uniaxial stretch and force-per-undeformed-area data (default is
            None).
        ps : array of shape (2, ...) or None, optional
            Experimental planar-shear stretch and force-per-undeformed-area data
            (default is None).
        bx : array of shape (2, ...) or None, optional
            Experimental biaxial stretch and force-per-undeformed-area data (default is
            None).
        incompressible : bool, optional
            A flag to enforce incompressible deformations (default is False).
        relative : bool, optional
            A flag to optimize relative instead of absolute residuals, i.e.
            ``(predicted - observed) / observed`` instead of ``predicted - observed``
            (default is False).
        **kwargs : dict, optional
            Optional keyword arguments are passed to
            :func:`scipy.optimize.least_squares`.

        Returns
        -------
        ConstitutiveMaterial
            A copy of the constitutive material with the optimized material parameters.
        scipy.optimize.OptimizeResult
            Represents the optimization result.


        Notes
        -----
        ..  warning::
            At least one load case, i.e. one of the arguments ``ux``, ``ps`` or ``bx``
            must not be ``None``.

        The vector of residuals is given in Eq. :eq:`material-optimize-residuals` in
        case of absolute residuals

        ..  math::
            :label: material-optimize-residuals

            \boldsymbol{r} &= \begin{bmatrix}
                \boldsymbol{r}_\text{ux} \\
                \boldsymbol{r}_\text{ps} \\
                \boldsymbol{r}_\text{bx}
            \end{bmatrix}

            r_\text{ux}(\lambda_i) &= P_\text{ux}(\lambda_i)
                - P_\text{ux, observed}(\lambda_i)

            r_\text{ps}(\lambda_i) &= P_\text{ps}(\lambda_i)
                - P_\text{ps, observed}(\lambda_i)

            r_\text{bx}(\lambda_i) &= P_\text{bx}(\lambda_i)
                - P_\text{bx, observed}(\lambda_i)

        and in Eq. :eq:`material-optimize-residuals-relative` in case of relative
        residuals.

        ..  math::
            :label: material-optimize-residuals-relative

            \boldsymbol{r} &= \begin{bmatrix}
                \boldsymbol{r}_\text{ux} \\
                \boldsymbol{r}_\text{ps} \\
                \boldsymbol{r}_\text{bx}
            \end{bmatrix}

            r_\text{ux}(\lambda_i)  &= \frac{
                P_\text{ux}(\lambda_i) - P_\text{ux, observed}(\lambda_i)}{
                    P_\text{ux, observed}(\lambda_i)
                }

            r_\text{ps}(\lambda_i)  &= \frac{
                P_\text{ps}(\lambda_i) - P_\text{ps, observed}(\lambda_i)}{
                    P_\text{ps, observed}(\lambda_i)
                }

            r_\text{bx}(\lambda_i)  &= \frac{
                P_\text{bx}(\lambda_i) - P_\text{bx, observed}(\lambda_i)}{
                    P_\text{bx, observed}(\lambda_i)
                }

        Examples
        --------
        The :func:`Extended Tube <felupe.extended_tube>` material model formulation is
        best-fitted on Treloar's uniaxial and biaxial tension data [1]_.

        ..  pyvista-plot::
            :context:

            >>> import numpy as np
            >>> import felupe as fem
            >>>
            >>> λ_ux, P_ux = np.array(
            ...     [
            ...         [1.000, 0.00],
            ...         [1.020, 0.26],
            ...         [1.125, 1.37],
            ...         [1.240, 2.30],
            ...         [1.390, 3.23],
            ...         [1.585, 4.16],
            ...         [1.900, 5.10],
            ...         [2.180, 6.00],
            ...         [2.420, 6.90],
            ...         [3.020, 8.80],
            ...         [3.570, 10.7],
            ...         [4.030, 12.5],
            ...         [4.760, 16.2],
            ...         [5.360, 19.9],
            ...         [5.750, 23.6],
            ...         [6.150, 27.4],
            ...         [6.400, 31.0],
            ...         [6.600, 34.8],
            ...         [6.850, 38.5],
            ...         [7.050, 42.1],
            ...         [7.150, 45.8],
            ...         [7.250, 49.6],
            ...         [7.400, 53.3],
            ...         [7.500, 57.0],
            ...         [7.600, 64.4],
            ...     ]
            ... ).T * np.array([[1.0], [0.0980665]])
            >>>
            >>> λ_bx, P_bx = np.array(
            >>>     [
            ...         [1.00, 0.00],
            ...         [1.03, 0.95],
            ...         [1.07, 1.60],
            ...         [1.12, 2.42],
            ...         [1.14, 2.62],
            ...         [1.20, 3.32],
            ...         [1.31, 4.43],
            ...         [1.42, 5.18],
            ...         [1.68, 6.60],
            ...         [1.94, 7.78],
            ...         [2.49, 9.79],
            ...         [3.03, 12.6],
            ...         [3.43, 14.7],
            ...         [3.75, 17.4],
            ...         [4.07, 20.1],
            ...         [4.26, 22.5],
            ...         [4.45, 24.7],
            ...     ]
            ... ).T * np.array([[1.0], [0.0980665]])
            >>>
            >>> umat = fem.Hyperelastic(fem.extended_tube)
            >>> umat_new, res = umat.optimize(
            ...     ux=[λ_ux, P_ux], bx=[λ_bx, P_bx], incompressible=True
            ... )
            >>>
            >>> ux = np.linspace(λ_ux.min(), λ_ux.max(), num=50)
            >>> bx = np.linspace(λ_bx.min(), λ_bx.max(), num=50)
            >>> ax = umat_new.plot(incompressible=True, ux=ux, bx=bx, ps=None)
            >>> ax.plot(λ_ux, P_ux, "C0x")
            >>> ax.plot(λ_bx, P_bx, "C1x")

        ..  pyvista-plot::
            :include-source: False
            :context:
            :force_static:

            >>> import pyvista as pv
            >>>
            >>> fig = ax.get_figure()
            >>> chart = pv.ChartMPL(fig)
            >>> chart.show()

        See Also
        --------
        scipy.optimize.least_squares : Solve a nonlinear least-squares problem with
            bounds on the variables.

        References
        ----------
        .. [1] L. R. G. Treloar, "Stress-strain data for vulcanised rubber under various
           types of deformation", Transactions of the Faraday Society, vol. 40. Royal
           Society of Chemistry (RSC), p. 59, 1944. doi:
           `10.1039/tf9444000059 <https://doi.org/10.1039/tf9444000059>`_. Data
           available at https://www.uni-due.de/mathematik/ag_neff/neff_hencky.
        """
        from scipy.optimize import least_squares

        experiments = []
        for lc in [ux, bx, ps]:
            experiment = (None, None)
            if lc is not None:
                experiment = np.asarray(lc).reshape(2, -1)
            experiments.append(experiment)

        # get sizes of material parameters and offsets as cumulative sum
        offsets = np.cumsum([np.asarray(y).size for y in self.kwargs.values()])[:-1]

        def fun(x):
            "Return the vector of residuals for given material parameters x."

            # update the material parameters
            for key, value in zip(self.kwargs.keys(), np.split(x, offsets)):
                self.kwargs[key] = value

            # evaluate the load cases by the material model formulation
            model = self.view(
                incompressible=incompressible,
                ux=experiments[0][0],
                bx=experiments[1][0],
                ps=experiments[2][0],
            ).evaluate()

            # calculate a list of residuals for each loadcase
            residuals = []
            for predicted, observed in zip(model, experiments):
                if observed[1] is not None:
                    res = predicted[1] - observed[1]
                    if relative:
                        observed_reference = np.array(observed[1])
                        observed_reference[observed_reference == 0] = 1
                        res /= observed_reference
                    residuals.append(res)

            return np.concatenate(residuals)

        # optimize the initial material parameters
        x0 = [np.asarray(value).ravel() for value in self.kwargs.values()]
        res = least_squares(fun=fun, x0=np.concatenate(x0), **kwargs)

        def std(hessian, residuals_variance):
            "Return the estimated errors (standard deviations) of parameters."
            return np.sqrt(np.diag(np.linalg.inv(hessian) * residuals_variance))

        # estimate the optimization errors for each material parameter
        hess = res.jac.T @ res.jac
        res.dx = std(hess, 2 * res.cost / (len(res.fun) - len(res.x)))

        # copy and update the material parameters of the material model formulation
        umat = copy(self)
        for key, value in zip(self.kwargs.keys(), np.split(res.x, offsets)):
            umat.kwargs[key] = value

        return umat, res

    def __and__(self, other_material):
        return CompositeMaterial(self, other_material)


def constitutive_material(Material, name=None):
    """A class-decorator for a constitutive material definition.

    Parameters
    ----------
    Material : object
        A class with methods for the gradient and the hessian of the strain energy
        density function per unit undeformed volume w.r.t. the deformation gradient
        tensor.
    name : str or None, optional
        The name of the derived class object. If None, the name is taken from
        ``Material`` (default is None).

    Returns
    -------
    object
        A derived class with multiple inheritance of
        :class:`~felupe.ConstitutiveMaterial` and ``Material``.

    Examples
    --------
    This example shows how to create a derived user material class to enable the
    methods from :class:`~felupe.ConstitutiveMaterial` on an externally defined user
    material.

    .. admonition:: This example requires external packages.
       :class: hint

       .. code-block::

          pip install matadi

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>> import matadi as mat
        >>>
        >>> MaterialHyperelastic = fem.constitutive_material(mat.MaterialHyperelastic)
        >>> umat = MaterialHyperelastic(mat.models.neo_hooke, C10=0.5)
        >>> ax = umat.plot(incompressible=True)

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    See Also
    --------
    felupe.ConstitutiveMaterial : Base class for constitutive material formulations.
    """
    if name is None:
        name = Material.__name__

    class ConstitutiveMaterialDerived(ConstitutiveMaterial, Material):
        pass

    ConstitutiveMaterialDerived.__name__ = name

    return ConstitutiveMaterialDerived


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

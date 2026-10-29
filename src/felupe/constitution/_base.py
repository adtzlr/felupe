# SPDX-License-Identifier: GPL-3.0-or-later

from copy import deepcopy as copy

import numpy as np

from ..math import dot, eigh, transpose
from ..math._chunks import CHUNKSIZE_POINTS
from ._chunks import evaluate_chunks
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

    Instead of the methods ``gradient(x)`` and ``hessian(x)``, a constitutive material
    may implement the methods ``_gradient(x)`` and ``_hessian(x)``. These are
    evaluated in chunks of cells by :meth:`gradient` and :meth:`hessian`, i.e. the
    arrays of the input list ``x`` are sliced along their last axis. The intermediate
    arrays of a chunk fit into the CPU caches, which speeds up the evaluation of large
    arrays. The number of evaluation points (quadrature points times cells) per chunk
    is given by the attribute ``chunksize`` (default is 8192, None evaluates all cells
    at once). The chunks are evaluated by a pool of threads if the attribute
    ``parallel`` is True (default is False). In this case, ``_gradient(x)`` and
    ``_hessian(x)`` must be thread-safe, i.e. intermediate results must be local
    variables (not attributes of the material). Optional output arrays (``out``) are
    handled by :meth:`gradient` and :meth:`hessian`.

    ..  code-block::

        class MyMaterialFormulation(fem.ConstitutiveMaterial):

            def __init__(self, parallel=False):
                self.x = [np.zeros((3, 3))]
                self.parallel = parallel

            def _gradient(self, x):
                "Gradients of the strain energy density function of a chunk of cells."

                F, p, statevars = x[0], x[1], x[-1]
                ...

                return [dWdF, dWdp, statevars_new]

            def _hessian(self, x):
                "Hessians of the strain energy density function of a chunk of cells."

                F, p, statevars = x[0], x[1], x[-1]
                ...

                return [d2WdFdF, d2WdFdp, d2Wdpdp]

    A time-dependent constitutive material, e.g. a viscoelastic material, takes the
    time increment :math:`\Delta t` as an input, like the deformation gradient. It is
    marked by the attribute ``time_dependent`` (default is False). Then, the time
    increment is a required keyword-argument ``dtime`` of :meth:`gradient` and
    :meth:`hessian`, e.g. ``umat.gradient(x, dtime=0.1)``, which is passed to
    ``_gradient(x, dtime)`` and ``_hessian(x, dtime)``, i.e. it is the same for all
    chunks of cells. A material which is not time-dependent does not take a time
    increment.

    ..  code-block::

        class MyViscousMaterialFormulation(fem.ConstitutiveMaterial):

            time_dependent = True

            def __init__(self):
                self.x = [np.zeros((3, 3))]

            def _gradient(self, x, dtime):
                "Gradients of the strain energy density function of a chunk of cells."

                F, statevars = x[0], x[-1]
                ...

                return [dWdF, statevars_new]

            def _hessian(self, x, dtime):
                "Hessians of the strain energy density function of a chunk of cells."

                F, statevars = x[0], x[-1]
                ...

                return [d2WdFdF]

    The same interface applies to material definitions which are not derived from
    :class:`~felupe.ConstitutiveMaterial`, e.g. of external packages: a material is
    time-dependent if it has an attribute ``time_dependent`` which is True. Then, its
    methods ``gradient(x, dtime)`` and ``hessian(x, dtime)`` take the time increment as
    keyword-argument ``dtime``. Without this attribute, a material is not time-dependent
    and the time increment is not passed to it, e.g. by the materials
    :class:`~felupe.NearlyIncompressible`, :class:`~felupe.ThreeFieldVariation` and
    :class:`~felupe.ThirdMediumContactMixed`, which take the time-dependency from their
    material, or by a composite material ``umat & other_umat``.

    See Also
    --------
    felupe.constitutive_material : A decorator for a constitutive material definition.
    """

    chunksize = CHUNKSIZE_POINTS
    parallel = False
    time_dependent = False

    def _check_time_increment(self, kwargs):
        """Check the time increment ``dtime`` of the keyword-arguments of
        :meth:`gradient` and :meth:`hessian`. A time-dependent material requires a time
        increment, a material which is not time-dependent does not take one."""

        if self.time_dependent:
            if kwargs.get("dtime") is None:
                raise ValueError(
                    f"The material {type(self).__name__} is time-dependent and requires "
                    "the time increment, e.g. `umat.gradient(x, dtime=0.1)`."
                )

        elif "dtime" in kwargs:
            message = (
                f"The material {type(self).__name__} is not time-dependent and does "
                "not take the time increment `dtime`."
            )

            if "dtime" in getattr(self, "kwargs", {}):
                message += (
                    " Its material parameter `dtime` is constant, i.e. the time "
                    "increment is not an input of the material."
                )

            raise TypeError(message)

    def gradient(self, x, *args, out=None, **kwargs):
        """Return the evaluated gradient(s) of the strain energy density function,
        evaluated by ``_gradient(x, *args, **kwargs)`` in chunks of cells.

        Parameters
        ----------
        x : list of ndarray
            The list with input arguments. These contain the extracted fields of a
            :class:`~felupe.FieldContainer` along with the old vector of state
            variables, ``[*field.extract(), statevars_old]``.
        *args : tuple, optional
            Optional arguments for ``_gradient()``.
        out : ndarray or None, optional
            A location into which the first gradient is stored (default is None).
        **kwargs : dict, optional
            Optional keyword-arguments for ``_gradient()``, e.g. the time increment
            ``dtime`` of a time-dependent material.

        Returns
        -------
        list of ndarray
            A list with the evaluated gradient(s) of the strain energy density function
            and the updated vector of state variables.

        Raises
        ------
        ValueError
            If the material is time-dependent and the time increment ``dtime`` is
            missing (or None).
        TypeError
            If the material is not time-dependent and a time increment ``dtime`` is
            given.
        """

        self._check_time_increment(kwargs)

        return evaluate_chunks(
            self._gradient,
            x,
            *args,
            chunksize=self.chunksize,
            parallel=self.parallel,
            out=out,
            **kwargs,
        )

    def hessian(self, x, *args, out=None, **kwargs):
        """Return the evaluated upper-triangle components of the hessian(s) of the
        strain energy density function, evaluated by ``_hessian(x, *args, **kwargs)``
        in chunks of cells.

        Parameters
        ----------
        x : list of ndarray
            The list with input arguments. These contain the extracted fields of a
            :class:`~felupe.FieldContainer` along with the old vector of state
            variables, ``[*field.extract(), statevars_old]``.
        *args : tuple, optional
            Optional arguments for ``_hessian()``.
        out : ndarray or None, optional
            A location into which the first hessian is stored (default is None).
        **kwargs : dict, optional
            Optional keyword-arguments for ``_hessian()``, e.g. the time increment
            ``dtime`` of a time-dependent material.

        Returns
        -------
        list of ndarray
            A list with the evaluated upper-triangle components of the hessian(s) of the
            strain energy density function.

        Raises
        ------
        ValueError
            If the material is time-dependent and the time increment ``dtime`` is
            missing (or None).
        TypeError
            If the material is not time-dependent and a time increment ``dtime`` is
            given.
        """

        self._check_time_increment(kwargs)

        return evaluate_chunks(
            self._hessian,
            x,
            *args,
            chunksize=self.chunksize,
            parallel=self.parallel,
            out=out,
            **kwargs,
        )

    def _gradient(self, x, *args, **kwargs):
        raise NotImplementedError(
            "A constitutive material must implement either `gradient(x)` or "
            "`_gradient(x)`, which is evaluated in chunks of cells."
        )

    def _hessian(self, x, *args, **kwargs):
        raise NotImplementedError(
            "A constitutive material must implement either `hessian(x)` or "
            "`_hessian(x)`, which is evaluated in chunks of cells."
        )

    def copy(self):
        "Return a deep-copy of the constitutive material."
        return copy(self)

    def _set_parameters(self, parameters):
        """Set the values of material parameters, i.e. of the keyword-arguments
        ``kwargs`` and of the attributes of the material with the same names.

        Parameters
        ----------
        parameters : dict
            A dict with the names and the new values of the material parameters.

        Notes
        -----
        The material parameters of a material may be stored in its keyword-arguments
        ``kwargs`` (e.g. of the materials with given functions like
        :class:`~felupe.Hyperelastic`) or as attributes with the same names (e.g. of
        :class:`~felupe.NeoHooke`). Materials which evaluate other quantities of their
        material parameters, e.g. other materials, have to update them by an
        overridden method.
        """

        for key, value in parameters.items():
            self.kwargs[key] = value

            if key in vars(self):
                setattr(self, key, value)

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
            :class:`~felupe.ViewMaterialIncompressible`, e.g. the load cases. The load
            cases of a time-dependent material include the (absolute) times, e.g.
            ``ux=[stretches, times]``.

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
        ux : array of shape (2, ...) or (3, ...) or None, optional
            Experimental uniaxial stretch and force-per-undeformed-area data (default is
            None). The (absolute) times are required as third row for a time-dependent
            material.
        ps : array of shape (2, ...) or (3, ...) or None, optional
            Experimental planar-shear stretch and force-per-undeformed-area data
            (default is None). The (absolute) times are required as third row for a
            time-dependent material.
        bx : array of shape (2, ...) or (3, ...) or None, optional
            Experimental biaxial stretch and force-per-undeformed-area data (default is
            None). The (absolute) times are required as third row for a time-dependent
            material.
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

        ..  note::
            For JAX-based materials, double-precision is required to optimize material
            parameters.

        The material parameters are the keyword-arguments of the material. They are
        set on a copy of the material, i.e. its keyword-arguments and its attributes
        with the same names, as well as the material parameters of its materials, e.g.
        of a composite material. The time increment is not a material parameter, i.e. a
        constant time increment ``dtime`` of the keyword-arguments is not optimized. The
        experimental data of a time-dependent material includes the (absolute) times
        of the data points, e.g. ``ux=[stretches, forces, times]``, which are taken for
        the time increments of the load case, see :class:`~felupe.ViewMaterial`.

            ..  code-block:: python

                import jax

                jax.config.update("jax_enable_x64", True)

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
        The :func:`Anssari-Benam Bucchi <felupe.anssari_benam_bucchi>` material model
        formulation is best-fitted on Treloar's uniaxial and biaxial tension data [1]_.

        ..  plot::

            >>> import numpy as np
            >>> import felupe as fem
            >>>
            >>> λ, P = np.array(
            ...     [
            ...         [1.000, 0.00],
            ...         [1.240, 2.30],
            ...         [1.585, 4.16],
            ...         [2.180, 6.00],
            ...         [3.020, 8.80],
            ...         [4.030, 12.5],
            ...         [4.760, 16.2],
            ...         [5.750, 23.6],
            ...         [6.850, 38.5],
            ...         [7.250, 49.6],
            ...         [7.600, 64.4],
            ...     ]
            ... ).T * np.array([[1.0], [0.0980665]])
            >>>
            >>> umat = fem.Hyperelastic(fem.anssari_benam_bucchi)
            >>> umat_new, res = umat.optimize(
            ...     ux=[λ, P], incompressible=True, relative=True
            ... )
            >>>
            >>> ux = np.linspace(λ.min(), λ.max(), num=50)
            >>> ax = umat_new.plot(incompressible=True, ux=ux, bx=None, ps=None)
            >>> lines = ax.plot(λ, P, "C0x")

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

        # the given load cases (load case and observed forces) in the order of the
        # evaluated load cases of the views, i.e. uniaxial, planar shear and biaxial
        experiments = {}
        for key, lc in zip(["ux", "ps", "bx"], [ux, ps, bx]):
            if lc is not None:
                data = np.asarray(lc, dtype=float)
                if data.ndim > 1 and len(data) == 3:
                    # stretches, forces and the times of a time-dependent material
                    stretches, forces, times = data.reshape(3, -1)
                    experiments[key] = ([stretches, times], forces)
                else:
                    stretches, forces = data.reshape(2, -1)
                    experiments[key] = (stretches, forces)

        if len(experiments) == 0:
            raise ValueError(
                "At least one load case is required, i.e. one of the arguments `ux`, "
                "`ps` or `bx` must not be None."
            )

        # the load cases of the views (without the load cases which are not given)
        load_cases = {"ux": None, "ps": None, "bx": None}
        load_cases.update({key: lc for key, (lc, forces) in experiments.items()})

        # get sizes of material parameters and offsets as cumulative sum (a constant
        # time increment is not a material parameter)
        keys = [key for key in self.kwargs.keys() if key != "dtime"]
        offsets = np.cumsum([np.asarray(self.kwargs[key]).size for key in keys])[:-1]

        def parameters(x):
            "Return a dict of material parameters for the vector of parameters x."
            return {
                key: value[0] if value.size == 1 else value
                for key, value in zip(keys, np.split(x, offsets))
            }

        # copy the material model formulation
        umat = self.copy()

        def fun(x):
            "Return the vector of residuals for given material parameters x."

            # update the material parameters
            umat._set_parameters(parameters(x))

            # evaluate the load cases by the material model formulation
            model = umat.view(incompressible=incompressible, **load_cases).evaluate()

            # calculate a list of residuals for each loadcase (same order)
            residuals = []
            for predicted, observed in zip(model, experiments.values()):
                active = ~np.isnan(observed[1])
                res = (predicted[1] - observed[1])[active]
                if relative:
                    observed_reference = np.array(observed[1])[active]
                    observed_reference[observed_reference == 0] = 1
                    res /= observed_reference
                residuals.append(res)

            return np.concatenate(residuals)

        # optimize the initial material parameters
        x0 = [np.asarray(umat.kwargs[key]).ravel() for key in keys]
        res = least_squares(fun=fun, x0=np.concatenate(x0), **kwargs)

        def std(hessian, residuals_variance):
            "Return the estimated errors (standard deviations) of parameters."
            return np.sqrt(np.diag(np.linalg.pinv(hessian) * residuals_variance))

        # estimate the optimization errors for each material parameter
        hess = res.jac.T @ res.jac
        res.dx = std(hess, 2 * res.cost / (len(res.fun) - len(res.x)))

        # save the final the material parameters
        umat._set_parameters(parameters(res.x))

        return umat, res

    def is_stable(self, x, hessian=None):
        """Return a boolean mask for stability of isotropic material model formulations.

        At a given deformation gradient, a normal force is applied on each principal
        stretch direction. If the resulting incremental stretches are positive, the
        material model formulation is considered to be stable at the given deformation
        gradient.

        Parameters
        ----------
        x : list of ndarray
            The list with input arguments. These contain the extracted fields of a
            :class:`~felupe.FieldContainer`.
        hessian : ndarray or None, optional
            Second partial derivative of the strain energy density function w.r.t. the
            deformation gradient. Default is None.

        Returns
        -------
        ndarray
            Boolean mask of stability.

        Notes
        -----

        ..  warning::

            This stability check will lead to a singular matrix for isotropic
            (hyperelastic) material model formulations without a volumetric part.

        Examples
        --------
        First, let's check the stability of the Neo-Hookean material model formulation.
        The stability is evaluated on (valid) principal stretches of a biaxial
        deformation. All deformations are stable.

        ..  plot::

            >>> import numpy as np
            >>> import felupe as fem
            >>>
            >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
            >>> view = umat.view()
            >>> λ = view.biaxial()[0]
            >>>
            >>> F = np.zeros((3, 3, 1, λ[0].size))
            >>> for a in range(3):
            ...     F[a, a] = λ[a]
            >>>
            >>> umat.is_stable([F])
            array([[ True,  True,  True,  True,  True,  True,  True,  True,  True,
                     True,  True,  True,  True,  True,  True,  True]])

        Now, let's check the stability of the Mooney-Rivlin material model formulation.
        The stability is evaluated on (valid) principal stretches of a biaxial
        deformation. Biaxial deformations are only stable up to a longitudinal stretch
        of 1.35.

        ..  plot::

            >>> import numpy as np
            >>> import felupe as fem
            >>> import felupe.constitution.tensortrax as mat
            >>>
            >>> umat = fem.Hyperelastic(
            ...     mat.models.hyperelastic.mooney_rivlin,
            ...     C10=0.25,
            ...     C01=0.25,
            ... ) & fem.Volumetric(bulk=5000)
            >>> view = umat.view()
            >>> λ = view.biaxial()[0]
            >>>
            >>> F = np.zeros((3, 3, 1, λ[0].size))
            >>> for a in range(3):
            ...     F[a, a] = λ[a]
            >>>
            >>> umat.is_stable([F])
            array([[ True,  True,  True,  True,  True,  True,  True,  True, False,
                    False, False, False, False, False, False, False]])

        """
        N = eigh(dot(transpose(x[0]), x[0]))[1]
        n = eigh(dot(x[0], transpose(x[0])))[1]

        if hessian is None:
            hessian = self.hessian(x)[0]

        A = np.einsum(
            "iJkL...,ia...,Ja...,kb...,Lb...->ab...",
            hessian,
            n,
            N,
            n,
            N,
            optimize=True,
        )

        stretch_change = np.diagonal(np.linalg.inv(A.T).T, axis1=0, axis2=1)

        return np.all(stretch_change > 0, axis=-1)

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
    methods from :class:`~felupe.ConstitutiveMaterial` on any (external) material.

    ..  plot::

        >>> import felupe as fem
        >>> import numpy as np
        >>>
        >>> class MyMaterialFormulation:
        ...     def __init__(self, a=5):
        ...         self.x = [np.zeros((3, 3))]
        ...         self.kwargs = {"a": a}
        ...
        ...     def gradient(self, x):
        ...         F, statevars = x[0], x[-1]
        ...         dWdF = self.kwargs["a"] * np.eye(3).reshape(3, 3, 1, 1)
        ...         return [dWdF, statevars]
        ...
        ...     def hessian(self, x, **kwargs):
        ...         F, statevars = x[0], x[-1]
        ...         d2WdFdF = self.kwargs["a"] * np.zeros((3, 3, 3, 3, 1, 1))
        ...         return [d2WdFdF]
        >>>
        >>> MyMaterial = fem.constitutive_material(MyMaterialFormulation)
        >>> umat = MyMaterial(a=0.5)
        >>> ax = umat.plot(incompressible=True)

    See Also
    --------
    felupe.ConstitutiveMaterial : Base class for constitutive material formulations.
    """
    if name is None:
        name = Material.__name__

    # the methods of the material, e.g. ``gradient()`` and ``hessian()``, take
    # precedence over the methods of the base class
    class ConstitutiveMaterialDerived(Material, ConstitutiveMaterial):
        pass

    ConstitutiveMaterialDerived.__name__ = name

    return ConstitutiveMaterialDerived


def _set_material_parameters(material, parameters):
    """Set the values of the material parameters of a material, which are included in
    its keyword-arguments (also for materials which are not derived from
    :class:`~felupe.ConstitutiveMaterial`)."""

    own = {key: value for key, value in parameters.items() if key in material.kwargs}

    if len(own) > 0:
        if hasattr(material, "_set_parameters"):
            material._set_parameters(own)
        else:
            material.kwargs.update(own)


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
    The gradients and the hessians are evaluated in chunks of cells, by a pool of
    threads if any of the materials is parallel.

    The composite material is time-dependent if any of the materials is time-dependent
    (attribute ``time_dependent``). The time increment ``dtime`` is only passed to the
    time-dependent materials, e.g. ``viscoelastic & fem.Volumetric(bulk=5000)``.

    ..  warning::
        Do not merge two constitutive materials with the same keys of material
        parameters. In this case, the values of these material parameters are taken from
        the first constitutive material.

    Examples
    --------
    ..  plot::

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

        # evaluate the chunks of cells in parallel, if any of the materials is parallel
        self.parallel = any(getattr(m, "parallel", False) for m in self.materials)

    @property
    def time_dependent(self):
        """A flag which is True if any of the materials is time-dependent. Then, the
        time increment ``dtime`` is passed to the time-dependent materials."""
        return any(getattr(m, "time_dependent", False) for m in self.materials)

    def _set_parameters(self, parameters):
        """Set the values of material parameters of the composite material and of the
        materials with these material parameters."""

        super()._set_parameters(parameters)

        for material in self.materials:
            _set_material_parameters(material, parameters)

    @staticmethod
    def _material_kwargs(material, kwargs):
        """Return the keyword-arguments for a material, i.e. without the time increment
        for a material which is not time-dependent."""

        if getattr(material, "time_dependent", False):
            return kwargs

        return {key: value for key, value in kwargs.items() if key != "dtime"}

    def _gradient(self, x, **kwargs):
        gradients = [
            material.gradient(x, **self._material_kwargs(material, kwargs))
            for material in self.materials
        ]
        nfields = max(1, len(x) - 1)
        P = [np.sum([grad[i] for grad in gradients], axis=0) for i in range(nfields)]
        statevars_new = gradients[0][-1]
        return [*P, statevars_new]

    def _hessian(self, x, **kwargs):
        hessians = [
            material.hessian(x, **self._material_kwargs(material, kwargs))
            for material in self.materials
        ]
        nfields = max(1, len(x) - 1)
        return [np.sum([hess[i] for hess in hessians], axis=0) for i in range(nfields)]

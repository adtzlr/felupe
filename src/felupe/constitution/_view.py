# SPDX-License-Identifier: GPL-3.0-or-later

import warnings
from copy import deepcopy

import numpy as np

from ..math import det, linsteps


def check_stretches(stretches):
    "Check if any stretch is lower or equal zero."

    if np.any(stretches <= 0.0):
        raise ValueError("All stretches must be greater than 0.")

    return


class PlotMaterial:
    "Plot force-stretch curves of constitutive material formulations."

    def _load_case(self, load_case):
        """Return the stretches and the time increments of a load case.

        Parameters
        ----------
        load_case : array_like
            The stretches of a load case or the stretches and the (absolute) times of a
            load case of a time-dependent material, ``[stretches, times]``.

        Returns
        -------
        stretches : ndarray
            The stretches of the load case.
        dtimes : ndarray or None
            The time increments of the load case (None for materials which are not
            time-dependent), i.e. the differences of the times with the initial time
            zero.
        """

        stretches = np.asarray(load_case, dtype=float)
        times = None

        if stretches.ndim == 2:
            if len(stretches) != 2:
                raise ValueError(
                    "A load case must be given by its stretches or by its stretches "
                    "and times, `[stretches, times]`."
                )
            stretches, times = stretches

        name = type(self.umat).__name__
        time_dependent = getattr(self.umat, "time_dependent", False)

        if time_dependent and times is None:
            raise ValueError(
                f"The material {name} is time-dependent and requires the (absolute) "
                "times of the load cases, e.g. `ux=[stretches, times]`."
            )

        if not time_dependent and times is not None:
            raise TypeError(
                f"The material {name} is not time-dependent and does not take the "
                "times of the load cases."
            )

        check_stretches(stretches)

        dtimes = None

        if times is not None:
            dtimes = np.diff(times, prepend=0.0)

            if np.any(dtimes < 0):
                raise ValueError(
                    "The times of a load case must not decrease (the initial time is "
                    "zero)."
                )

        return stretches, dtimes

    def _evaluate_path(self, F, statevars=None, dtimes=None):
        """Return the first Piola-Kirchhoff stress tensors for the deformation gradients
        of the increments of a load case (on the last axis) and the state variables of
        the last increment.

        Materials with state variables and time-dependent materials are evaluated
        increment by increment, i.e. the updated state variables of an increment are
        the old state variables of the next increment and each increment takes its time
        increment. Other materials are evaluated for all increments at once.
        """

        time_dependent = getattr(self.umat, "time_dependent", False)

        if not (self.statevars_included or time_dependent):
            P, statevars = self.umat.gradient([F, None])
            return P, statevars

        if statevars is None:
            statevars = np.zeros((*self.umat.x[-1].shape, 1, 1))

        P = np.zeros_like(F)

        for increment in range(F.shape[-1]):
            kwargs = {} if dtimes is None else {"dtime": dtimes[increment]}
            P[..., [increment]], statevars = self.umat.gradient(
                [F[..., [increment]], statevars], **kwargs
            )

        return P, statevars

    def evaluate(self):
        """Evaluate normal force per undeformed area vs stretch curves for the
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

        initial_statevars = deepcopy(self.statevars)
        if self.ux is not None:
            data.append(self.uniaxial())
            self.statevars = initial_statevars

        if self.ps is not None:
            data.append(self.planar())
            self.statevars = initial_statevars

        if self.bx is not None:
            data.append(self.biaxial())
            self.statevars = initial_statevars

        return data

    def plot(self, ax=None, show_title=True, show_kwargs=True, **kwargs):
        """Plot normal force per undeformed area vs stretch curves for the elementary
        homogeneous incompressible deformations uniaxial tension/compression, planar
        shear and biaxial tension."""

        import matplotlib.pyplot as plt

        data = self.evaluate()

        if ax is None:
            fig, ax = plt.subplots()

        for stretches, force, label in data:
            ax.plot(stretches[0], force, label=label, **kwargs)

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
                p = []
                for key, value in self.umat.kwargs.items():
                    if value is None:
                        # optional arguments which are not set are not shown
                        continue
                    elif isinstance(value, str):
                        val = f'"{value}"'
                    elif callable(value):
                        name = [v.title() for v in value.__name__.split("_")]
                        val = f'"{" ".join(name)}"'
                    elif np.ndim(value) > 0:
                        val = " ".join([f"{v:.3g}" for v in np.ravel(value)])
                        if np.size(value) > 1:
                            val = f"[{val}]"
                    else:
                        val = f"{value:.3g}"
                    p.append(f"{key}={val}")
                parameters = ", ".join(p)

                ax.set_title(parameters, fontdict=dict(fontsize="small"), wrap=True)

        return ax


class ViewMaterial(PlotMaterial):
    """Create views on normal force per undeformed area vs stretch curves for the
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
        ``linsteps([0.7, 2.5], num=36)``. The (absolute) times are required as second
        row for a time-dependent material, e.g. ``[stretches, times]``.
    ps : ndarray, optional
        Array with stretches for planar shear. Default is
        ``linsteps([1.0, 2.5], num=30)``. The (absolute) times are required as second
        row for a time-dependent material, e.g. ``[stretches, times]``.
    bx : ndarray, optional
        Array with stretches for equi-biaxial tension. Default is
        ``linsteps([1.0, 1.75], num=15)```. The (absolute) times are required as second
        row for a time-dependent material, e.g. ``[stretches, times]``.
    statevars : ndarray or None, optional
        Array with state variables (default is None). If None, the state variables are
        assumed to be initially zero.

    Notes
    -----
    The load cases of a time-dependent material, i.e. a material with an attribute
    ``time_dependent`` which is True, include the (absolute) times of the stretches,
    e.g. ``ux=[stretches, times]``. The time increments are the differences of the
    times, with the initial time zero (like a :class:`~felupe.TimeStep`), and they are
    passed to the material for each increment of a load case. Each load case starts at
    the initial time with the initial state variables. Holding times are given by
    repeated stretches.

    Examples
    --------
    ..  plot::
        :context: close-figs

        >>> import felupe as fem
        >>>
        >>> umat = fem.OgdenRoxburgh(fem.NeoHooke(mu=1, bulk=2), r=3, m=1, beta=0)
        >>> view = fem.ViewMaterial(
        ...     umat,
        ...     ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
        ...     ps=None,
        ...     bx=None,
        ... )
        >>> ax = view.plot(show_title=True, show_kwargs=True)

    The load case of a time-dependent material includes the times, e.g. a loading with
    a holding time at the maximum stretch and an unloading with a lower stretch rate.

    ..  plot::
        :context: close-figs

        >>> umat = fem.MaterialStrain(
        ...     material=fem.linear_elastic_viscoelastic,
        ...     λ=2.0,
        ...     μ=1.0,
        ...     G=[3.0, 2.0],
        ...     τ=[1.0, 10.0],
        ...     statevars=((2, 3, 3),),
        ... )
        >>> stretches = fem.math.linsteps([1, 1.1, 1.1, 1], num=[10, 10, 10])
        >>> times = fem.math.linsteps([0, 1, 11, 31], num=[10, 10, 10])
        >>> view = fem.ViewMaterial(umat, ux=[stretches, times], ps=None, bx=None)
        >>> ax = view.plot(show_title=True, show_kwargs=True)

    """

    def __init__(
        self,
        umat,
        ux=linsteps([0.7, 2.5], num=36),
        ps=linsteps([1, 2.5], num=30),
        bx=linsteps([1, 1.75], num=15),
        statevars=None,
    ):
        self.umat = umat
        self.ux = ux
        self.ps = ps
        self.bx = bx
        self.statevars_included = np.prod(self.umat.x[-1].shape) > 0
        self.statevars = statevars

    def uniaxial(self, stretches=None, **kwargs):
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

        stretches, dtimes = self._load_case(stretches)

        λ1 = stretches

        λ2 = λ3 = 1 / np.sqrt(λ1)
        eye = np.eye(3).reshape(3, 3, 1, 1)

        def fun(λ3):
            λ2 = λ3
            F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)
            P, statevars = self._evaluate_path(F, self.statevars, dtimes)
            return P[2, 2].ravel()

        from scipy.optimize import root

        res = root(fun, λ3, **kwargs)

        if not res.success:
            λ2 = λ3 = np.ones_like(λ1)
            res = root(fun, λ3, **kwargs)

            if not res.success:
                raise ValueError("Uniaxial deformation failed.")

        λ2 = λ3 = res.x
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, self.statevars = self._evaluate_path(F, self.statevars, dtimes)

        valid = det(F) > np.sqrt(np.finfo(float).eps)
        if not np.all(valid):
            warnings.warn("Uniaxial data with volume ratio det(F) <= 0 included.")
            P[0, 0][~valid] = np.nan

        return (λ1, λ2, λ3), P[0, 0].ravel(), "Uniaxial"

    def planar(self, stretches=None, **kwargs):
        """Normal force per undeformed area vs stretch curve for a planar shear
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
            stretches = self.ps

        stretches, dtimes = self._load_case(stretches)

        λ1 = stretches
        λ2 = np.ones_like(λ1)
        λ3 = 1 / λ1
        eye = np.eye(3).reshape(3, 3, 1, 1)

        def fun(λ3):
            F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)
            P, statevars = self._evaluate_path(F, self.statevars, dtimes)
            return P[2, 2].ravel()

        from scipy.optimize import root

        res = root(fun, λ3, **kwargs)

        if not res.success:
            λ3 = np.ones_like(λ1)
            res = root(fun, λ3, **kwargs)

            if not res.success:
                raise ValueError("Planar deformation failed.")

        λ3 = res.x
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, self.statevars = self._evaluate_path(F, self.statevars, dtimes)

        valid = det(F) > np.sqrt(np.finfo(float).eps)
        if not np.all(valid):
            warnings.warn("Planar Shear data with volume ratio det(F) <= 0 included.")
            P[0, 0][~valid] = np.nan

        return (λ1, λ2, λ3), P[0, 0].ravel(), "Planar Shear"

    def biaxial(self, stretches=None, **kwargs):
        """Normal force per undeformed area vs stretch curve for a equi-biaxial
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
            stretches = self.bx

        stretches, dtimes = self._load_case(stretches)

        λ1 = λ2 = stretches
        λ3 = 1 / λ1**2
        eye = np.eye(3).reshape(3, 3, 1, 1)

        def fun(λ3):
            F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)
            P, statevars = self._evaluate_path(F, self.statevars, dtimes)
            return P[2, 2].ravel()

        from scipy.optimize import root

        res = root(fun, λ3, **kwargs)

        if not res.success:
            λ3 = np.ones_like(λ1)
            res = root(fun, λ3, **kwargs)

            if not res.success:
                raise ValueError("Biaxial deformation failed.")

        λ3 = res.x
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, self.statevars = self._evaluate_path(F, self.statevars, dtimes)

        valid = det(F) > np.sqrt(np.finfo(float).eps)
        if not np.all(valid):
            warnings.warn("Biaxial data with volume ratio det(F) <= 0 included.")
            P[0, 0][~valid] = np.nan

        return (λ1, λ2, λ3), P[0, 0].ravel(), "Biaxial"


class ViewMaterialIncompressible(PlotMaterial):
    """Create views on normal force per undeformed area vs stretch curves for the
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
        ``linsteps([0.7, 2.5], num=36)``. The (absolute) times are required as second
        row for a time-dependent material, e.g. ``[stretches, times]``.
    ps : ndarray, optional
        Array with stretches for incompressible planar shear. Default is
        ``linsteps([1, 2.5], num=30)``. The (absolute) times are required as second
        row for a time-dependent material, e.g. ``[stretches, times]``.
    bx : ndarray, optional
        Array with stretches for incompressible equi-biaxial tension. Default is
        ``linsteps([1, 1.75], num=15)``. The (absolute) times are required as second
        row for a time-dependent material, e.g. ``[stretches, times]``.
    statevars : ndarray or None, optional
        Array with state variables (default is None). If None, the state variables are
        assumed to be initially zero.

    Notes
    -----
    The load cases of a time-dependent material, i.e. a material with an attribute
    ``time_dependent`` which is True, include the (absolute) times of the stretches,
    e.g. ``ux=[stretches, times]``. The time increments are the differences of the
    times, with the initial time zero (like a :class:`~felupe.TimeStep`), and they are
    passed to the material for each increment of a load case. Each load case starts at
    the initial time with the initial state variables. Holding times are given by
    repeated stretches.

    Examples
    --------
    ..  plot::
        :context: close-figs

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.extended_tube, Gc=0.2, Ge=0.2, beta=0.2, delta=0.1)
        >>> preview = fem.ViewMaterialIncompressible(umat)
        >>> ax = preview.plot(show_title=True, show_kwargs=True)

    ..  plot::
        :context: close-figs

        >>> umat = fem.OgdenRoxburgh(fem.NeoHooke(mu=1), r=3, m=1, beta=0)
        >>> view = fem.ViewMaterialIncompressible(
        ...     umat,
        ...     ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
        ...     ps=None,
        ...     bx=None,
        ... )
        >>> ax = view.plot(show_title=True, show_kwargs=True)

    The load case of a time-dependent material includes the times, e.g. cycles with
    increasing maximum stretches at a constant stretch rate.

    ..  plot::
        :context: close-figs

        >>> umat = fem.Hyperelastic(
        ...     fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, nstatevars=6
        ... )
        >>> stretches = fem.math.linsteps([1, 1.5, 1, 2, 1], num=[5, 5, 10, 10])
        >>> times = fem.math.linsteps([0, 0.5, 1, 2, 3], num=[5, 5, 10, 10])
        >>> view = fem.ViewMaterialIncompressible(
        ...     umat, ux=[stretches, times], ps=None, bx=None
        ... )
        >>> ax = view.plot(show_title=True, show_kwargs=True)

    """

    def __init__(
        self,
        umat,
        ux=linsteps([0.7, 2.5], num=36),
        ps=linsteps([1, 2.5], num=30),
        bx=linsteps([1, 1.75], num=15),
        statevars=None,
    ):
        self.umat = umat
        self.ux = ux
        self.ps = ps
        self.bx = bx
        self.statevars_included = np.prod(self.umat.x[-1].shape) > 0
        self.statevars = statevars

    def uniaxial(self, stretches=None):
        """Normal force per undeformed area vs stretch curve for a uniaxial
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

        stretches, dtimes = self._load_case(stretches)

        λ1 = stretches
        λ2 = λ3 = 1 / np.sqrt(λ1)
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, self.statevars = self._evaluate_path(F, self.statevars, dtimes)

        return (
            (λ1, λ2, λ3),
            (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel(),
            "Uniaxial (Incompressible)",
        )

    def planar(self, stretches=None):
        """Normal force per undeformed area vs stretch curve for a planar shear
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

        stretches, dtimes = self._load_case(stretches)

        λ1 = stretches
        λ2 = np.ones_like(λ1)
        λ3 = 1 / λ1
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, self.statevars = self._evaluate_path(F, self.statevars, dtimes)

        return (
            (λ1, λ2, λ3),
            (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel(),
            "Planar Shear (Incompressible)",
        )

    def biaxial(self, stretches=None):
        """Normal force per undeformed area vs stretch curve for a equi-biaxial
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

        stretches, dtimes = self._load_case(stretches)

        λ1 = λ2 = stretches
        λ3 = 1 / λ1**2
        eye = np.eye(3).reshape(3, 3, 1, 1)
        F = eye * np.array([λ1, λ2, λ3]).reshape(1, 3, 1, -1)

        P, self.statevars = self._evaluate_path(F, self.statevars, dtimes)

        return (
            (λ1, λ2, λ3),
            (P[0, 0] - λ3 / λ1 * P[2, 2]).ravel(),
            "Biaxial (Incompressible)",
        )

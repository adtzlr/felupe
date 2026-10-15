# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import numpy as np

from ..assembly import IntegralForm
from ..constitution import AreaChange
from ..field import FieldAxisymmetric
from ..math import ddot, dya
from ._helpers import (
    Assemble,
    Evaluate,
    Results,
    StateNearlyIncompressible,
    cauchy_stress,
    kirchhoff_stress,
    mass_matrix,
)
from ._solidbody import Solid


class SolidBodyNearlyIncompressible(Solid):
    r"""A (nearly) incompressible solid body with methods for the assembly of sparse
    vectors/matrices. The constitutive material definition must provide the distortional
    part of the strain energy density function per unit undeformed volume only. The
    volumetric part of the strain energy density function is automatically added on
    additional internal (dual) cell-wise constant fields.

    Parameters
    ----------
    umat : class
        A class which provides methods for evaluating the gradient and the hessian of
        the isochoric part of the strain energy density function per unit undeformed
        volume :math:`\hat\psi(\boldsymbol{F})`. The function signatures must be
        ``dψdF, ζ_new = umat.gradient([F, ζ])`` for the gradient and
        ``d2ψdFdF = umat.hessian([F, ζ])`` for the hessian of
        the strain energy density function :math:`\hat{\psi}(\boldsymbol{F})`, where
        :math:`\boldsymbol{F}` is the deformation gradient tensor and :math:`\zeta`
        holds the array of internal state variables.
    field : FieldContainer
        A field container with one or more fields.
    bulk : float or None, optional
        The bulk modulus :math:`K` of the volumetric part of the strain energy density
        function :math:`U(\bar{J})`, see ``volumetric_kind``. It is required for the
        kinds ``"linear"`` and ``"log"`` and it is ignored for ``"custom"``. Default is
        None.
    state : StateNearlyIncompressible or None, optional
        A valid initial state for a (nearly) incompressible solid (default is None).
    statevars : ndarray or None, optional
        Array of initial internal state variables (default is None).
    density : float or None, optional
        The density of the solid body.
    volumetric_kind : str, optional
        The kind of the volumetric part of the strain energy density function
        :math:`U(\bar{J})`: ``"linear"`` for :math:`U(\bar{J})=K(\bar{J}-1)^2/2` with
        a linear relation of the pressure and the volume ratio, ``"log"`` for
        :math:`U(\bar{J})=K\ln(\bar{J})^2/2` or ``"custom"`` with the callables
        ``volumetric_gradient`` and ``volumetric_hessian``, see Eq.
        :eq:`nearlyincompressible-volumetric`. Default is ``"linear"``.
    volumetric_gradient : callable or None, optional
        The first derivative of the volumetric part of the strain energy density
        function w.r.t. the volume ratio :math:`U'(\bar{J})`, which is required for
        ``volumetric_kind="custom"`` (otherwise, it is ignored). It is called with the
        array of the cell-wise volume ratios and must return the array of the cell-wise
        values. Default is None.
    volumetric_hessian : callable or None, optional
        The second derivative of the volumetric part of the strain energy density
        function w.r.t. the volume ratio :math:`U''(\bar{J})`, which is required for
        ``volumetric_kind="custom"`` (otherwise, it is ignored). It is called with the
        array of the cell-wise volume ratios and must return the array of the cell-wise
        values or a scalar. Default is None.

    Notes
    -----
    ..  note::

        This nearly-incompressible solid body works best with
        :class:`~felupe.RegionQuad` and :class:`~felupe.RegionHexahedron` regions.
        :class:`~felupe.RegionQuadraticTriangle` and
        :class:`~felupe.RegionQuadraticTetra` regions are also supported.

    The total potential energy of internal forces for a three-field variational
    approach, suitable for nearly-incompressible material behaviour, is given in Eq.
    :eq:`nearlyincompressible`

    ..  math::
        :label: nearlyincompressible

        \Pi_{int}(\boldsymbol{F}, p, \bar{J}) =
            \int_V \hat{\psi}(\boldsymbol{F})\ dV +
            \int_V U(\bar{J})\ dV +
            \int_V p (J - \bar{J})\ dV

    with the volumetric part of the strain energy density function :math:`U(\bar{J})`,
    see Eq. :eq:`nearlyincompressible-volumetric`. Its first and second derivatives
    w.r.t. the volume ratio are abbreviated by :math:`\bar{U}'=U'(\bar{J})` and
    :math:`\bar{U}''=U''(\bar{J})`. The variations of the total potential energy are
    given in Eq. :eq:`nearlyinc-variations`

    ..  math::
        :label: nearlyinc-variations

        \delta_\boldsymbol{u}(\Pi_{int}) &=
            \int_V \left( \frac{\partial \hat{\psi}}{\partial \boldsymbol{F}} +
            p\ \frac{\partial J}{\partial \boldsymbol{F}} \right) :
            \delta\boldsymbol{F}\ dV
        \longrightarrow \boldsymbol{r}_\boldsymbol{u}

        \delta_p(\Pi_{int}) &=
            \int_V \left( J - \bar{J} \right)\ \delta p\ dV
        \longrightarrow r_p

        \delta_\bar{J}(\Pi_{int}) &=
            \int_V \left( \bar{U}' - p \right)\ \delta \bar{J}\ dV
        \longrightarrow r_{\bar{J}}

    and linearizations, see Eq. :eq:`nearlyinc-linearizations` [1]_ [2]_ [3]_. The
    right-arrows in Eq. :eq:`nearlyinc-variations` and Eq.
    :eq:`nearlyinc-linearizations` represent the assembly into system scalars, vectors
    or matrices.

    ..  math::
        :label: nearlyinc-linearizations

        \delta_\boldsymbol{u}\Delta_\boldsymbol{u}(\Pi_{int}) &=
            \int_V \delta\boldsymbol{F} : \left(
                \frac{\partial^2 \hat{\psi}}{
                    \partial \boldsymbol{F}\ \partial \boldsymbol{F}
                } +
                p \frac{\partial^2 J}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}
            \right) : \Delta\boldsymbol{F}\ dV
        \longrightarrow \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}}


        \delta_\boldsymbol{u}\Delta_p(\Pi_{int}) &= \int_V \delta \boldsymbol{F} :
            \frac{\partial J}{\partial \boldsymbol{F}}\ \Delta p\ dV
        \longrightarrow \boldsymbol{K}_{\boldsymbol{u}p}

        \delta_p\Delta_{\bar{J}}(\Pi_{int}) &=
            \int_V \delta p\ (-1)\ \Delta \bar{J}\ dV
        \longrightarrow -V

        \delta_{\bar{J}}\Delta_{\bar{J}}(\Pi_{int}) &=
            \int_V \delta \bar{J}\ \bar{U}''\ \Delta \bar{J}\ dV
        \longrightarrow \bar{U}''\ V

    The assembled constraint equations for the variations w.r.t. the dual fields
    :math:`p` and :math:`\bar{J}` are given in Eq. :eq:`nearlyinc-constraints`.

    ..  math::
        :label: nearlyinc-constraints

        r_p &= \left( \frac{v}{V} - \bar{J} \right) V

        r_{\bar{J}} &= \left( \bar{U}' - p \right) V


    The volumetric part of the strain energy density function is selected by
    ``volumetric_kind``. It is denoted in Eq. :eq:`nearlyincompressible-volumetric`
    along with its first and second derivatives for ``volumetric_kind="linear"``
    (default)

    ..  math::
        :label: nearlyincompressible-volumetric

        U(\bar{J}) &= \frac{K}{2} \left( \bar{J} - 1 \right)^2

        U'(\bar{J}) &= K \left( \bar{J} - 1 \right)

        U''(\bar{J}) &= K

    and in Eq. :eq:`nearlyincompressible-volumetric-log` for
    ``volumetric_kind="log"``.

    ..  math::
        :label: nearlyincompressible-volumetric-log

        U(\bar{J}) &= \frac{K}{2} \ln \left( \bar{J} \right)^2

        U'(\bar{J}) &= K\ \frac{\ln \left( \bar{J} \right)}{\bar{J}}

        U''(\bar{J}) &= K\ \frac{1 - \ln \left( \bar{J} \right)}{\bar{J}^2}

    For ``volumetric_kind="custom"``, the first and second derivatives are given by the
    callables ``volumetric_gradient`` and ``volumetric_hessian``. The undeformed
    configuration is free of stress only if :math:`U'(1)=0`, and a positive second
    derivative :math:`U''(\bar{J})>0` is required for a stable volumetric response.
    Note that the second derivative of the logarithmic kind is negative for
    :math:`\bar{J}>e`.


    **Hu-Washizu Three-Field-Variational Principle**

    The Three-Field-Variation :math:`(\boldsymbol{u},p,\bar{J})` leads to a linearized
    equation system with nine sub block-matrices, see Eq. :eq:`hu-washizu` [4]_.
    Due to the fact that the equation system is derived by a potential, the matrix is
    symmetric and hence, only six independent sub-matrices have to be evaluated.
    Furthermore, by the application of the mean dilatation technique, two of the
    remaining six sub-matrices are identified to be zero. That means four sub-matrices
    are left to be evaluated, where two non-zero sub-matrices are scalar-valued entries.

    ..  math::
        :label: hu-washizu

        \begin{bmatrix}
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}}
                & \boldsymbol{K}_{\boldsymbol{u}p} & \boldsymbol{0} \\
            \boldsymbol{K}_{\boldsymbol{u}p}^T & 0 & -V \\
            \boldsymbol{0}^T & -V  & \bar{U}''\ V
        \end{bmatrix} \cdot \begin{bmatrix}
            \delta \boldsymbol{u} \\
            \delta p \\
            \delta \bar{J}
        \end{bmatrix} + \begin{bmatrix}
            \boldsymbol{r}_\boldsymbol{u} \\
            r_p \\
            r_\bar{J}
        \end{bmatrix} =
        \begin{bmatrix}
            \boldsymbol{0}\\
            0 \\
            0
        \end{bmatrix}

    A condensed representation of the equation system, only dependent on the primary
    unknowns :math:`\boldsymbol{u}` is carried out. To do so, the second line is
    multiplied by the second derivative :math:`\bar{U}''`, see Eq.
    :eq:`nearlyinc-condensed`.

    ..  math::
        :label: nearlyinc-condensed

        \begin{bmatrix}
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}}
                & \boldsymbol{K}_{\boldsymbol{u}p} & \boldsymbol{0} \\
            \bar{U}'' \boldsymbol{K}_{\boldsymbol{u}p}^T & 0 & -\bar{U}''\ V \\
            \boldsymbol{0}^T & -V  & \bar{U}''\ V
        \end{bmatrix} \cdot \begin{bmatrix}
            \delta \boldsymbol{u} \\
            \delta p \\
            \delta \bar{J}
        \end{bmatrix} + \begin{bmatrix}
            \boldsymbol{r}_\boldsymbol{u} \\
            \bar{U}''\ r_p \\
            r_\bar{J}
        \end{bmatrix} =
        \begin{bmatrix}
            \boldsymbol{0}\\
            0 \\
            0
        \end{bmatrix}

    Lines two and three are contracted by summation as given in Eq.
    :eq:`nearlyinc-contracted-sum`. This eliminates :math:`\bar{J}` from the unknowns.

    ..  math::
        :label: nearlyinc-contracted-sum

        \begin{bmatrix}
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}}
                & \boldsymbol{K}_{\boldsymbol{u}p} \\
            \bar{U}'' \boldsymbol{K}_{\boldsymbol{u}p}^T & -V
        \end{bmatrix} \cdot \begin{bmatrix}
            \delta \boldsymbol{u} \\
            \delta p
        \end{bmatrix} + \begin{bmatrix}
            \boldsymbol{r}_\boldsymbol{u} \\
            \bar{U}''\ r_p + r_\bar{J}
        \end{bmatrix} =
        \begin{bmatrix}
            \boldsymbol{0}\\
            0
        \end{bmatrix}

    Next, the second line is left-expanded by
    :math:`\frac{1}{V}~\boldsymbol{K}_{\boldsymbol{u}p}` and both equations are summed
    up again, see :eq:`nearlyinc-contracted-sum-final`.

    ..  math::
        :label: nearlyinc-contracted-sum-final

        \left(
            \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}}
                + \frac{\bar{U}''}{V}~\boldsymbol{K}_{\boldsymbol{u}p} \otimes
                    \boldsymbol{K}_{\boldsymbol{u}p}
        \right) \cdot \delta \boldsymbol{u} +
        \boldsymbol{r}_\boldsymbol{u} + \frac{\bar{U}''~r_p + r_\bar{J}}{V}
            \boldsymbol{K}_{\boldsymbol{u}p} = \boldsymbol{0}

    The secondary unknowns are evaluated after solving the primary unknowns, see
    Eq. :eq:`nearlyinc-final`.

    ..  math::
        :label: nearlyinc-final

        \delta \bar{J} &= \frac{1}{V} \delta \boldsymbol{u} \cdot
            \boldsymbol{K}_{\boldsymbol{u}p} + \frac{v}{V} - \bar{J}

        \delta p &= \bar{U}' + \bar{U}''\ \delta \bar{J} - p

    The condensed constraint equation in Eq. :eq:`nearlyinc-contracted-sum-final` is
    given in Eq. :eq:`nearlyinc-constraint`

    ..  math::
        :label: nearlyinc-constraint

        \frac{\bar{U}''~r_p + r_{\bar{J}}}{V} =
            \bar{U}' + \bar{U}'' \left( \frac{v}{V} - \bar{J} \right) - p

    and the deformed volume is evaluated by Eq. :eq:`nearlyinc-deformed-volume`.

    ..  math::
        :label: nearlyinc-deformed-volume

        v = \int_V J\ dV

    The terms :math:`\bar{U}' + \bar{U}''\ (\dots)` in Eq. :eq:`nearlyinc-final` and
    Eq. :eq:`nearlyinc-constraint` are the linearizations of
    :math:`U'(\bar{J} + \delta \bar{J})` and :math:`U'(v/V)`. Instead, their exact
    values are used, i.e. the pressure is updated by :math:`p = U'(\bar{J})` and the
    condensed constraint is evaluated by :math:`U'(v/V) - p`. Both are identical for
    the linear kind. Hence, the assembled vector is the gradient of the total
    potential energy of internal forces of the mean dilatation technique, see Eq.
    :eq:`nearlyinc-mean-dilatation`, where the sum is taken over the cells.

    ..  math::
        :label: nearlyinc-mean-dilatation

        \Pi_{int}(\boldsymbol{u}) = \int_V \hat{\psi}(\boldsymbol{F})\ dV +
            \sum V\ U\left( \frac{v}{V} \right)


    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
        >>>
        >>> umat = fem.NeoHooke(mu=1)
        >>> solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
        >>>
        >>> table = fem.math.linsteps([0, 1], num=5)
        >>> step = fem.Step(
        ...     items=[solid],
        ...     ramp={boundaries["move"]: table},
        ...     boundaries=boundaries,
        ... )
        >>>
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> solid.plot("Principal Values of Cauchy Stress").show()

    References
    ----------
    ..  [1] J. Bonet, A. J. Gil, and R. D. Wood, "Nonlinear Solid Mechanics for Finite
        Element Analysis: Statics". Cambridge University Press, Jun. 05, 2016.
        doi: 10.1017/cbo9781316336144.

    ..  [2] O. C. Zienkiewicz, R. L. Taylor and J.Z. Zhu, "The Finite Element Method:
        its Basis and Fundamentals". Elsevier, 2013. doi: 10.1016/c2009-0-24909-9.

    ..  [3] G. A. Holzapfel, "Nonlinear Solid Mechanics. A Continuum Approach for
        Engineering". Wiley, 2000. isbn: 047182304X.

    ..  [4] J. A. Schönherr, P. Schneider, and C. Mittelstedt, "Robust hybrid/mixed
        finite elements for rubber-like materials under severe compression",
        Computational Mechanics, vol. 70, no. 1. Springer Science and Business Media
        LLC, pp. 101–122, Mar. 31, 2022. doi: 10.1007/s00466-022-02157-y.

    See Also
    --------
    felupe.StateNearlyIncompressible : A State with internal cell-wise constant fields
        for (nearly) incompressible solid bodies.
    """

    def __init__(
        self,
        umat,
        field,
        bulk=None,
        state=None,
        statevars=None,
        density=None,
        volumetric_kind="linear",
        volumetric_gradient=None,
        volumetric_hessian=None,
    ):
        self.umat = umat
        self.field = field
        self.bulk = bulk
        self.density = density

        # gradient U'(J) and hessian U''(J) of
        # volumetric strain energy density function U(J)
        self.volumetric_kind = volumetric_kind
        self.volumetric_gradient = volumetric_gradient
        self.volumetric_hessian = volumetric_hessian

        kinds = ["linear", "log", "custom"]

        if volumetric_kind not in kinds:
            raise ValueError(
                f'Unknown volumetric_kind="{volumetric_kind}". Available kinds are '
                + ", ".join(f'"{kind}"' for kind in kinds)
                + "."
            )

        if volumetric_kind == "custom":
            if not (callable(volumetric_gradient) and callable(volumetric_hessian)):
                raise ValueError(
                    'volumetric_kind="custom" requires the callables '
                    "volumetric_gradient and volumetric_hessian."
                )

        elif bulk is None:
            raise ValueError(
                f'volumetric_kind="{volumetric_kind}" requires the bulk modulus, '
                "e.g. bulk=5000."
            )

        _dUdJ_dict = {
            "custom": volumetric_gradient,
            "linear": lambda J: self.bulk * (J - 1),
            "log": lambda J: self.bulk * np.log(J) / J,
        }
        _d2UdJdJ_dict = {
            "custom": volumetric_hessian,
            "linear": lambda J: self.bulk,
            "log": lambda J: self.bulk * (1 - np.log(J)) / (J**2),
        }

        self.dUdJ = _dUdJ_dict[self.volumetric_kind]
        self.d2UdJdJ = _d2UdJdJ_dict[self.volumetric_kind]

        self._area_change = AreaChange()
        self._form = IntegralForm

        # volume of undeformed configuration
        if isinstance(self.field[0], FieldAxisymmetric):
            R = self.field[0].radius
            dA = self.field.region.dV
            dV = 2 * np.pi * R * dA
        else:
            dV = self.field.region.dV
        self.V = dV.sum(0)

        self.results = Results(stress=True, elasticity=True)

        if statevars is not None:
            self.results.statevars = statevars
        else:
            statevars_shape = (0,)
            if hasattr(umat, "x"):
                statevars_shape = umat.x[-1].shape
            self.results.statevars = np.zeros(
                (
                    *statevars_shape,
                    field.region.quadrature.npoints,
                    field.region.mesh.ncells,
                ),
                dtype=self.field[0].values.dtype,
            )

        if state is None:
            # init state of internal fields
            self.results.state = StateNearlyIncompressible(field)
        else:
            self.results.state = state

        self.results.kinematics = self._extract(self.field)
        self.assemble = Assemble(
            vector=self._vector, matrix=self._matrix, mass=self._mass
        )

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
            kirchhoff_stress=self._kirchhoff_stress,
        )

    def checkpoint(self):
        """Return a checkpoint of the nearly-incompressible solid body.

        Returns
        -------
        dict
            A dict with checkpoint arrays / objects.

        Examples
        --------
        ..  pyvista-plot::
            :force_static:

            import felupe as fem

            mesh = fem.Rectangle(b=(3, 1), n=(7, 3))
            region = fem.RegionQuad(mesh)
            field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

            umat = fem.NeoHooke(mu=1)
            solid = fem.SolidBodyNearlyIncompressible(umat=umat, field=field, bulk=5000)

            # 1. vertical compression
            boundaries = fem.dof.uniaxial(
                field, clamped=True, sym=False, axis=1, return_loadcase=False
            )
            ramp = {boundaries["move"]: fem.math.linsteps([0, -0.2], num=2)}
            step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
            job = fem.Job(steps=[step]).evaluate()

            # checkpoint the current state of deformation
            checkpoint = solid.checkpoint()

            # 2a. horizontal shear (right)
            boundaries = fem.dof.shear(
                field, sym=False, moves=(0, 0, -0.2), return_loadcase=False
            )
            ramp = {boundaries["move"]: fem.math.linsteps([0, 1], num=2)}
            step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
            job = fem.Job(steps=[step]).evaluate()

            # (1.) restore compression without shear
            solid.restore(checkpoint)

            # 2b. horizontal shear (left)
            boundaries = fem.dof.shear(
                field, sym=False, moves=(0, 0, -0.2), return_loadcase=False
            )
            ramp = {boundaries["move"]: fem.math.linsteps([0, -1], num=2)}
            step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
            job = fem.Job(steps=[step]).evaluate()

        See Also
        --------
        felupe.SolidBody.restore : Restore a checkpoint of a nearly-incompressible solid
            body inplace.
        """

        return {
            **self.field.checkpoint(),
            "results.state.u": self.results.state.u.copy(),
            "results.state.p": self.results.state.p.copy(),
            "results.state.J": self.results.state.J.copy(),
            "results.statevars": self.results.statevars.copy(),
        }

    def restore(self, checkpoint, restore_statevars=True, restore_state=True):
        r"""Restore a checkpoint inplace.

        Parameters
        ----------
        checkpoint : dict
            A dict with checkpoint arrays / objects.
        restore_statevars : bool, optional
            Flag to restore state variables. This is a power feature and must be used
            with caution! Default is True.
        restore_state : bool, optional
            Flag to restore the internal fields, i.e. the pressure :math:`p` and the
            volume ratio :math:`\bar{J}`, of the checkpoint. If False, the internal
            fields are re-initialized by the deformation of the restored field,
            :math:`\bar{J} = v / V` and :math:`p = U'(\bar{J})`. Default is True.

        Notes
        -----
        The deformation gradient is evaluated for the restored field before the
        results are re-evaluated. Hence, the internal fields are not updated by the
        deformation gradient of the last evaluation (e.g. of a failed Newton
        iteration).

        See Also
        --------
        felupe.SolidBodyNearlyIncompressible.checkpoint : Return a checkpoint of the
            nearly-incompressible solid body.
        """

        self.field.restore(checkpoint)

        # deformation gradient of the restored field. The internal fields must not be
        # updated w.r.t. the deformation gradient of the last evaluation.
        state = self.results.state
        self.results.kinematics = state.F = self.field.extract(
            out=self.results.kinematics
        )

        if restore_state:
            state.u[:] = checkpoint["results.state.u"]
            state.p[:] = checkpoint["results.state.p"]
            state.J[:] = checkpoint["results.state.J"]
        else:
            state.u[:] = self.field[0].values
            state.J[:] = state.volume() / self.V
            state.p[:] = self.dUdJ(state.J)

        if restore_statevars:
            self.results.statevars[:] = checkpoint["results.statevars"]

        # results must be re-evaluated (without an update of the internal fields)
        self.evaluate.gradient()
        self.evaluate.hessian()

        # reset force and stiffness
        self.results.force = None
        self.results.stiffness = None

    def revolve(self, n=11, phi=180, x0=None):
        """Return a revolved solid body.

        Parameters
        ----------
        n : int, optional
            Number of n-point revolutions (or (n-1) cell revolutions), default is 11.
        phi : float or ndarray, optional
            Revolution angle in degree (default is 180).
        x0 : FieldContainer or None, optional
            The revolved top-level field container. Default is None.

        Returns
        -------
        SolidBody
            The revolved solid body.

        See Also
        --------
        SolidBody.revolve : Return a revolved solid body

        """

        new_field = self.field.revolve(n=n, phi=phi)
        if hasattr(self.field, "x0"):
            if x0 is None:
                raise ValueError(
                    "The revolved top-level field container must be provided as the"
                    " x0-argument."
                )
            new_field.x0 = x0

        # create a new solid body
        new_solid = SolidBodyNearlyIncompressible(
            umat=self.umat,
            field=new_field,
            density=self.density,
            bulk=self.bulk,
            volumetric_kind=self.volumetric_kind,
            volumetric_gradient=self.volumetric_gradient,
            volumetric_hessian=self.volumetric_hessian,
        )

        # expand state variables (no rotation is applied here)
        new_shape = np.array(new_solid.results.statevars.shape)
        old_shape = np.array(self.results.statevars.shape)

        # ratio of shape of new vs. old state variables
        reps = np.zeros(len(new_shape), dtype=int)
        reps[0] = new_shape[0]
        reps[1:] = new_shape[1:] // old_shape[1:]

        new_solid.results.statevars[:] = np.tile(self.results.statevars, reps)

        # expand internal field variables (displacement)
        new_shape_internal = np.array(new_solid.results.state.p.shape)
        old_shape_internal = np.array(self.results.state.p.shape)

        # ratio of shape of new vs. old internal field variables
        reps_internal = new_shape_internal // old_shape_internal

        new_solid.results.state.u[:] = new_field[0].values
        new_solid.results.state.p[:] = np.tile(self.results.state.p, reps_internal)
        new_solid.results.state.J[:] = np.tile(self.results.state.J, reps_internal)

        return new_solid

    def _vector(
        self, field=None, parallel=False, items=None, args=(), kwargs=None, block=True
    ):
        if kwargs is None:
            kwargs = {}

        self.results.stress = self._gradient(
            field, parallel=parallel, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.stress,
            v=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.integrate_shape_function_gradient(
            parallel=parallel, out=self.results._force_values
        )
        v = self.results.state.volume()
        p = self.results.state.p

        constraint = np.multiply(h, self.dUdJ(v / self.V) - p, out=h)

        self.results.force_values = form.integrate(
            parallel=parallel, out=self.results.force_values
        )
        np.add(
            self.results.force_values[0], constraint, out=self.results.force_values[0]
        )

        self.results.force = form.assemble(
            values=self.results.force_values, block=block
        )

        return self.results.force

    def _matrix(
        self, field=None, parallel=False, items=None, args=(), kwargs=None, block=True
    ):
        if kwargs is None:
            kwargs = {}

        self.results.elasticity = self._hessian(
            field, parallel=parallel, args=args, kwargs=kwargs
        )

        form = self._form(
            fun=self.results.elasticity,
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        h = self.results.state.integrate_shape_function_gradient(
            parallel=parallel, out=self.results._force_values
        )
        H = dya(h, h, out=self.results._stiffness_values)
        bulk_H = np.multiply(H, self.d2UdJdJ(self.results.state.J), out=H)
        constraint = np.divide(bulk_H, self.V, out=bulk_H)

        self.results.stiffness_values = form.integrate(
            parallel=parallel, out=self.results.stiffness_values
        )
        np.add(
            self.results.stiffness_values[0],
            constraint,
            out=self.results.stiffness_values[0],
        )

        self.results.stiffness = form.assemble(
            values=self.results.stiffness_values, block=block
        )

        return self.results.stiffness

    def _extract(self, field, parallel=False):
        u = field[0].values
        u0 = self.results.state.u
        h = self.results.state.integrate_shape_function_gradient(
            parallel=parallel, out=self.results._force_values
        )
        v = self.results.state.volume()

        du = (u - u0)[field.region.mesh.cells].transpose([1, 2, 0])

        self.field = field
        self.results.kinematics = self.results.state.F = self.field.extract(
            out=self.results.kinematics
        )

        # update state variables
        self.results.state.J[:] = (ddot(du, h, mode=(2, 2)) + v) / self.V
        self.results.state.p[:] = self.dUdJ(self.results.state.J)
        self.results.state.u[:] = u

        return self.results.kinematics

    def _gradient(self, field=None, parallel=False, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

        dJdF = self._area_change.function
        F = self.results.kinematics[0]
        statevars = self.results.statevars

        p = self.results.state.p

        if "out" in inspect.signature(self.umat.gradient).parameters:
            kwargs["out"] = self.results.gradient

        [self.results.gradient, self.results._statevars] = self.umat.gradient(
            [F, statevars], *args, **kwargs
        )
        self.results.stress = [
            np.add(self.results.gradient, p * dJdF([F])[0], out=self.results.gradient)
        ]

        return self.results.stress

    def _hessian(self, field=None, parallel=False, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.results.kinematics = self._extract(field, parallel=parallel)

        d2JdF2 = self._area_change.gradient
        F = self.results.kinematics[0]
        statevars = self.results.statevars
        p = self.results.state.p

        if "out" in inspect.signature(self.umat.hessian).parameters:
            kwargs["out"] = self.results.hessian

        self.results.hessian = self.umat.hessian([F, statevars], *args, **kwargs)[0]
        self.results.elasticity = [
            np.add(self.results.hessian, p * d2JdF2([F])[0], out=self.results.hessian)
        ]

        return self.results.elasticity

    def _kirchhoff_stress(self, field=None):
        self._gradient(field)

        # the stress tensor of the formulation and the deformation gradient tensor
        P = self.results.stress[0]
        F = self.results.kinematics[0]

        return kirchhoff_stress(P, F)

    def _cauchy_stress(self, field=None):
        self._gradient(field)

        # the stress tensor of the formulation and the deformation gradient tensor
        P = self.results.stress[0]
        F = self.results.kinematics[0]

        return cauchy_stress(P, F)

    def _mass(self, density=None):
        if density is None:
            density = self.density

        return mass_matrix(self.field, density, form=self._form)

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

import inspect
import warnings

import numpy as np

from ..assembly import IntegralForm
from ..math import det, dot, transpose
from ..view import ViewSolid
from ._helpers import Assemble, Evaluate, Results


class Solid:
    "Base class for solid bodies which provides methods for visualisations."

    def view(
        self,
        point_data=None,
        cell_data=None,
        cell_type=None,
        project=None,
        stress_type="Cauchy",
    ):
        """View the solid with optional given dicts of point- and cell-data items.

        Parameters
        ----------
        point_data : dict or None, optional
            Additional point-data dict (default is None).
        cell_data : dict or None, optional
            Additional cell-data dict (default is None).
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).
        project : callable or None, optional
            Project stress at quadrature-points to mesh-points (default is None).
        stress_type : str or None, optional
            The type of stress which is exported, either "Cauchy", "Kirchhoff" or None.
            If None, the first Piola-Kirchhoff stress (engineering stress in linear
            elasticity) is used. Default is "Cauchy".

        Returns
        -------
        felupe.ViewSolid
            An object which provides visualization methods for solid bodies,
            e.g. :class:`felupe.SolidBody`.

        See Also
        --------
        felupe.ViewSolid : Visualization methods for :class:`felupe.SolidBody`.
        felupe.project: Project given values at quadrature-points to mesh-points.
        felupe.topoints: Shift given values at quadrature-points to mesh-points.
        """

        return ViewSolid(
            self.field,
            solid=self,
            point_data=point_data,
            cell_data=cell_data,
            cell_type=cell_type,
            project=project,
            stress_type=stress_type,
        )

    def plot(self, *args, project=None, **kwargs):
        """Plot the solid body.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        felupe.project: Project given values at quadrature-points to mesh-points.
        felupe.topoints: Shift given values at quadrature-points to mesh-points.
        """

        stress_type = "Cauchy"

        if len(args) > 0:
            name = kwargs.pop("name", args[0])

            if "Stress" in name:
                stress_type = (
                    name.lower()
                    .split("principal values of ")[-1]
                    .split("equivalent of ")[-1]
                    .split("stress")[0]
                    .rstrip()
                )

        if len(stress_type) == 0:
            stress_type = None

        return self.view(project=project, stress_type=stress_type).plot(*args, **kwargs)

    def screenshot(
        self,
        *args,
        filename="solidbody.png",
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the solid body.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )

    def imshow(self, *args, ax=None, dpi=None, **kwargs):
        """Take a screenshot of the solid body, show the image data in a figure and
        return the ax.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(dpi=dpi)

        ax.imshow(self.screenshot(*args, filename=None, **kwargs))
        ax.set_axis_off()

        return ax


class SolidBody(Solid):
    r"""A SolidBody with methods for the assembly of sparse vectors/matrices.

    Parameters
    ----------
    umat : class
        A class which provides methods for evaluating the gradient and the hessian of
        the strain energy density function per unit undeformed volume. The function
        signatures must be ``dψdF, ζ_new = umat.gradient([F, ζ])`` for
        the gradient and ``d2ψdFdF = umat.hessian([F, ζ])`` for the hessian of
        the strain energy density function :math:`\psi(\boldsymbol{F})`, where
        :math:`\boldsymbol{F}` is the deformation gradient tensor and :math:`\zeta`
        holds the array of internal state variables.
    field : FieldContainer
        A field container with one or more fields.
    statevars : ndarray or None, optional
        Array of initial internal state variables (default is None).
    density : float or None, optional
        The density of the solid body.
    block : bool, optional
        Assemble a sparse matrix from sparse sub-blocks or assemble a sparse vector by
        stacking sparse matrices vertically (row wise). Default is True.
    apply : callable or None, optional
        Apply a callable on the assembled vectors and sparse matrices. Default is None.
    multiplier : float or None, optional
        A scale factor for the assembled vector and matrix. Default is None.

    Notes
    -----
    The total potential energy of internal forces is given in Eq. :eq:`solidbody`

    ..  math::
        :label: solidbody

        \Pi_{int}(\boldsymbol{F}) = \int_V \psi(\boldsymbol{F})\ dV

    with its variation, see Eq. :eq:`solidbody-variation`

    ..  math::
        :label: solidbody-variation

        \delta_\boldsymbol{u}(\Pi_{int}) =
            \int_V \frac{\partial \psi}{\partial \boldsymbol{F}} \ dV
        \longrightarrow \boldsymbol{f}_\boldsymbol{u}

    and linearization, see Eq. :eq:`solidbody-linearization`. The right-arrows in
    Eq. :eq:`solidbody-variation` and Eq. :eq:`solidbody-linearization`
    represent the assembly into system scalars, vectors or matrices.

    ..  math::
        :label: solidbody-linearization

        \delta_\boldsymbol{u}\Delta_\boldsymbol{u}(\Pi_{int}) =
            \int_V \delta\boldsymbol{F} :
                \frac{\partial^2 \psi}{
                    \partial \boldsymbol{F}\ \partial \boldsymbol{F}
                } : \Delta\boldsymbol{F}\ dV
        \longrightarrow \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}}

    The displacement-based formulation leads to a linearized equation system as given in
    Eq. :eq:`solidbody-final`.

    ..  math::
        :label: solidbody-final

        \boldsymbol{K}_{\boldsymbol{u}\boldsymbol{u}} \cdot \delta \boldsymbol{u} +
            \boldsymbol{f}_\boldsymbol{u} = \boldsymbol{0}

    ..  note::
        This class also supports ``umat`` with mixed-field formulations like
        :class:`~felupe.NearlyIncompressible` or :class:`~felupe.ThreeFieldVariation`.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
        >>>
        >>> umat = fem.NeoHooke(mu=1, bulk=2)
        >>> solid = fem.SolidBody(umat, field)
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

    See Also
    --------
    felupe.SolidBodyNearlyIncompressible : A (nearly) incompressible solid body with
        methods for the assembly of sparse vectors/matrices.
    """

    def __init__(
        self,
        umat,
        field,
        statevars=None,
        density=None,
        block=True,
        apply=None,
        multiplier=None,
    ):
        self.umat = umat
        self.field = field
        self.density = density
        self.block = block
        self.apply = apply

        self.results = Results(stress=True, elasticity=True)
        self.results.kinematics = self._extract(self.field)

        if statevars is not None:
            # state variables are located in the results
            self.results.statevars = statevars

        else:
            # init the shape of the state variables
            statevars_shape = (0,)

            # take the shape of the last item of the list if the provided umat has an
            # x-attribute
            if hasattr(umat, "x"):
                statevars_shape = umat.x[-1].shape

            # create and fill the array of state variables with zeros
            # state variables are located in the results
            self.results.statevars = np.zeros(
                (
                    *statevars_shape,
                    field.region.quadrature.npoints,
                    field.region.mesh.ncells,
                )
            )

        self.assemble = Assemble(
            vector=self._vector,
            matrix=self._matrix,
            mass=self._mass,
            multiplier=multiplier,
        )

        self.evaluate = Evaluate(
            gradient=self._gradient,
            hessian=self._hessian,
            cauchy_stress=self._cauchy_stress,
            kirchhoff_stress=self._kirchhoff_stress,
        )

    def checkpoint(self):
        """Return a checkpoint of the solid body.

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

            umat = fem.NeoHooke(mu=1, bulk=2)
            solid = fem.SolidBody(umat=umat, field=field)

            # 1. vertical compression
            boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, sym=False, axis=1)
            ramp = {boundaries["move"]: fem.math.linsteps([0, -0.2], num=2)}
            step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
            job = fem.Job(steps=[step]).evaluate()

            # checkpoint the current state of deformation
            checkpoint = solid.checkpoint()

            # 2a. horizontal shear (right)
            boundaries, loadcase = fem.dof.shear(field, sym=False, moves=(0, 0, -0.2))
            ramp = {boundaries["move"]: fem.math.linsteps([0, 1], num=2)}
            step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
            job = fem.Job(steps=[step]).evaluate()

            # (1.) restore compression without shear
            solid.restore(checkpoint)

            # 2b. horizontal shear (left)
            boundaries, loadcase = fem.dof.shear(field, sym=False, moves=(0, 0, -0.2))
            ramp = {boundaries["move"]: fem.math.linsteps([0, -1], num=2)}
            step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
            job = fem.Job(steps=[step]).evaluate()

        See Also
        --------
        felupe.SolidBody.restore : Restore a checkpoint of a solid body inplace.
        """

        return {
            **self.field.checkpoint(),
            "results.statevars": self.results.statevars.copy(),
        }

    def restore(self, checkpoint, restore_statevars=True):
        """Restore a checkpoint inplace.

        Parameters
        ----------
        checkpoint : dict
            A dict with checkpoint arrays / objects.
        restore_statevars : bool, optional
            Flag to restore state variables. This is a power feature and must be used
            with caution! Default is True.

        See Also
        --------
        felupe.SolidBody.checkpoint : Return a checkpoint of the solid body.
        """

        self.field.restore(checkpoint)
        if restore_statevars:
            self.results.statevars[:] = checkpoint["results.statevars"]

        # results must be re-evaluated
        self.evaluate.gradient(self.field)
        self.evaluate.hessian(self.field)

        # reset force and stiffness
        self.results.force = None
        self.results.stiffness = None

    def revolve(self, n=11, phi=180):
        """Return a revolved solid body.

        Parameters
        ----------
        n : int, optional
            Number of n-point revolutions (or (n-1) cell revolutions), default is 11.
        phi : float or ndarray, optional
            Revolution angle in degree (default is 180).

        Returns
        -------
        SolidBody
            The revolved solid body.

        Examples
        --------
        First, create an axisymmetric model. A non-homogeneous uniaxial tension load
        case is applied on the solid body.

        ..  pyvista-plot::
            :context:
            :force_static:

            >>> import felupe as fem
            >>>
            >>> region = fem.RegionQuad(mesh=fem.Rectangle(n=6))
            >>> field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])
            >>>
            >>> umat = fem.NeoHookeCompressible(mu=1, lmbda=2)
            >>> solid = fem.SolidBody(umat=umat, field=field)
            >>>
            >>> boundaries, loadcase = fem.dof.uniaxial(
            ...     solid.field, clamped=True, sym=False
            ... )
            >>> step = fem.Step(items=[solid], boundaries=boundaries)
            >>> job = fem.Job(steps=[step]).evaluate()
            >>>
            >>> solid.plot("Principal Values of Cauchy Stress").show()

        The solid body is now revolved around the x-axis. This model may now be used to
        apply non-axisymmetric loads. Here, the same load case is also applied on the
        3d-model.

        ..  pyvista-plot::
            :context:
            :force_static:

            >>> new_solid = solid.revolve(n=11, phi=180)
            >>> boundaries, loadcase = fem.dof.uniaxial(
            ...     new_solid.field, clamped=True, sym=(0, 0, 1)
            ... )
            >>>
            >>> step = fem.Step(items=[new_solid], boundaries=boundaries)
            >>> job = fem.Job(steps=[step]).evaluate()
            >>>
            >>> new_solid.plot("Principal Values of Cauchy Stress").show()

        See Also
        --------
        SolidBodyNearlyIncompressible.revolve : Return a revolved solid body

        """

        new_field = self.field.revolve(n=n, phi=phi)

        # create a new solid body
        new_solid = SolidBody(
            umat=self.umat,
            field=new_field,
            density=self.density,
            block=self.block,
            apply=self.apply,
            multiplier=self.assemble.multiplier,
        )

        # expand state variables (no rotation is applied here)
        new_shape = np.array(new_solid.results.statevars.shape)
        old_shape = np.array(self.results.statevars.shape)

        reps = np.zeros(len(new_shape), dtype=int)
        reps[0] = new_shape[0]
        reps[1:] = new_shape[1:] // old_shape[1:]

        new_solid.results.statevars[:] = np.tile(self.results.statevars, reps)

        return new_solid

    def _vector(
        self,
        field=None,
        parallel=False,
        items=None,
        args=(),
        kwargs=None,
        block=None,
        apply=None,
    ):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.field = field

        if block is None:
            block = self.block

        if apply is None:
            apply = self.apply

        # evaluate the (first Piola-Kirchhoff) stress tensor and store it in the results
        self.results.stress = self._gradient(field, args=args, kwargs=kwargs)

        # assemble the internal force vector
        # optionally, use only the first n items for mixed-field formulations
        self.results.force = IntegralForm(
            fun=self.results.stress[slice(items)],
            v=self.field,
            dV=self.field.region.dV,
        ).assemble(parallel=parallel, block=block)

        # apply a callback on the assembled internal force vector
        if apply is not None:
            self.results.force = apply(self.results.force)

        return self.results.force

    def _matrix(
        self,
        field=None,
        parallel=False,
        items=None,
        args=(),
        kwargs=None,
        block=None,
        apply=None,
    ):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.field = field

        if block is None:
            block = self.block

        if apply is None:
            apply = self.apply

        # evaluate the fourth-order elasticity tensor and store it in the results
        # (associated to the first Piola-Kirchhoff stress tensor, i.e. the partial
        # derivative of the first Piola-Kirchhoff stress tensor w.r.t. the deformation
        # gradient tensor)
        self.results.elasticity = self._hessian(field, args=args, kwargs=kwargs)

        # assemble the (sparse) tangent stiffness matrix
        # optionally, use only the first n items for mixed-field formulations
        form = IntegralForm(
            fun=self.results.elasticity[slice(items)],
            v=self.field,
            u=self.field,
            dV=self.field.region.dV,
        )

        # in a first step, integrate the weak-form and store the stiffness values
        # (this dense array is only allocated once and will be re-used in all following
        # evaluations)
        self.results.stiffness_values = form.integrate(
            parallel=parallel, out=self.results.stiffness_values
        )

        # finally, the dense array of stiffness-values is assembled into the sparse
        # matrix
        self.results.stiffness = form.assemble(
            values=self.results.stiffness_values, block=block
        )

        # apply a callback on the assembled tangent stiffness matrix
        if apply is not None:
            self.results.stiffness = apply(self.results.stiffness)

        return self.results.stiffness

    def _extract(self, field):
        "Evaluate and return the kinematics (the deformation gradient tensor)."

        self.field = field
        self.results.kinematics = self.field.extract(out=self.results.kinematics)

        return self.results.kinematics

    def _gradient(self, field=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # update the deformation gradient
        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(self.field)

        if "out" in inspect.signature(self.umat.gradient).parameters:
            kwargs["out"] = self.results.gradient

        # evaluate the partial derivative of the strain energy density w.r.t. the
        # deformation gradient
        gradient = self.umat.gradient(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )
        # store the first Piola-Kirchhoff stress tensor in the results
        self.results.gradient = gradient[0]

        # store all gradients and the temporary state variables in the results
        # the state variables are only updated to the public results-attribute if the
        # solution is valid
        self.results.stress, self.results._statevars = gradient[:-1], gradient[-1]

        return self.results.stress

    def _hessian(self, field=None, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if field is not None:
            self.field = field
            self.results.kinematics = self._extract(self.field)

        if "out" in inspect.signature(self.umat.hessian).parameters:
            kwargs["out"] = self.results.hessian

        # evaluate the second partial derivative of the strain energy density w.r.t. the
        # deformation gradient
        self.results.elasticity = self.umat.hessian(
            [*self.results.kinematics, self.results.statevars], *args, **kwargs
        )

        # store the partial derivative of the first Piola-Kirchhoff stress tensor w.r.t.
        # the deformation gradient in the results
        self.results.hessian = self.results.elasticity[0]

        return self.results.elasticity

    def _kirchhoff_stress(self, field=None):
        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]

        return dot(P, transpose(F))

    def _cauchy_stress(self, field=None):
        self._gradient(field)

        P = self.results.stress[0]
        F = self.results.kinematics[0]

        if P.shape[:2] == (2, 2):
            warnings.warn(
                "\n".join(
                    [
                        "Cauchy stress tensor can't be evaluated on a 2d-Field.",
                        "Falling-back to the Kirchhoff stress tensor.",
                    ]
                )
            )
            J = 1
        else:
            J = det(F)

        return dot(P, transpose(F)) / J

    def _mass(self, density=None):
        if density is None:
            density = self.density

        field = self.field[0].as_container()
        dim = field[0].dim

        form = IntegralForm(
            fun=[density * np.eye(dim).reshape(dim, dim, 1, 1)],
            v=field,
            u=field,
            dV=field.region.dV,
            grad_v=[False],
            grad_u=[False],
        )

        return form.assemble()

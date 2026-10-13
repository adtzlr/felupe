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

from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
from scipy.sparse import coo_matrix

from .._helpers import Assemble, Results

ContactMultipliers = namedtuple("ContactMultipliers", ["normal", "tangential"])

ContactTrial = namedtuple(
    "ContactTrial", ["contact", "force", "stiffness", "dx", "dx_ref", "active", "slip"]
)


def _assemble_vector(dof, dof_center, forces, size, dtype):
    "Assemble the forces (m, dim) of the points and the reaction of the center-point."

    rows = np.concatenate([dof.ravel(), dof_center.ravel()])
    data = np.concatenate([forces.ravel(), -forces.sum(axis=0)])
    cols = np.zeros_like(rows)

    return coo_matrix((data, (rows, cols)), shape=(size, 1), dtype=dtype).tocsr()


def _assemble_matrix(dof, dof_center, blocks, size, dtype):
    """Assemble the stiffness blocks (m, dim, dim) of the points, which are connected
    to the center-point."""

    m, dim, _ = blocks.shape
    center = np.broadcast_to(dof_center.ravel(), (m, dim))

    rows, cols, data = [], [], []
    for row, col, sign in [
        (dof, dof, 1),
        (dof, center, -1),
        (center, dof, -1),
        (center, center, 1),
    ]:
        rows.append(np.broadcast_to(row[:, :, None], blocks.shape).ravel())
        cols.append(np.broadcast_to(col[:, None, :], blocks.shape).ravel())
        data.append(sign * blocks.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    return coo_matrix((data, (rows, cols)), shape=(size, size), dtype=dtype).tocsr()


class ContactRigidBase(ABC):
    r"""A base class for a node-to-surface contact, where the surface is given by a
    rigid body which is connected to a center-point.

    Notes
    -----
    The geometry of the rigid surface is defined by the method ``_kinematics(dx)`` of a
    subclass. For given gap vectors :math:`\Delta \boldsymbol{x}` between the points and
    the center-point, it returns the gaps :math:`g`, the outward normal vectors
    :math:`\boldsymbol{n} = \partial g / \partial \Delta \boldsymbol{x}` and the
    derivatives of the normal vectors w.r.t. the gap vectors (the curvatures)
    :math:`\boldsymbol{H} = \partial \boldsymbol{n} / \partial \Delta \boldsymbol{x}`,
    or None for a flat surface. The rigid surface does not rotate, i.e. only the
    translational degrees of freedom of the center-point are coupled.

    The base class provides the penalty-based normal contact, the Coulomb friction with
    the return mapping from the converged state, the stiffness-based multipliers, the
    checkpoints and the assembly of the sparse vector and matrix. With curvatures, the
    stiffness matrix includes the derivatives of the normal vectors, see
    :class:`~felupe.ContactRigidCylinder`.

    See Also
    --------
    felupe.ContactRigidPlane : A node-to-surface contact with a rigid plane.
    felupe.ContactRigidCylinder : A node-to-surface contact with a rigid cylinder.
    """

    def __init__(
        self,
        field,
        points,
        centerpoint,
        friction=0.0,
        multiplier=1.0,
        multiplier_tangential=0.1,
        items=None,
        multiplier_update="increment",
        symmetric=False,
        elastic_slip=None,
        curvature="converged",
    ):
        self.field = field
        self.mesh = field.region.mesh
        self.points = np.array(points)
        self.centerpoint = centerpoint

        if len(self.points) == 0:
            self.points = self.points.astype(int)

        if self.points.dtype == bool:
            self.points = np.where(self.points)[0]

        if self.centerpoint < 0:
            self.centerpoint = self.mesh.npoints + self.centerpoint

        if self.centerpoint in self.mesh.points_without_cells:
            self.mesh.points_without_cells = self.mesh.points_without_cells[
                self.mesh.points_without_cells != self.centerpoint
            ]

        self.friction = friction
        self.items = items

        updates = ["increment", "iteration", "once"]
        if multiplier_update not in updates:
            raise ValueError(f"The multiplier update must be one of {updates}.")

        self.multiplier = multiplier
        self.multiplier_tangential = multiplier_tangential
        self.multiplier_update = multiplier_update
        self.multipliers = None
        self.symmetric = symmetric

        if elastic_slip is not None and not elastic_slip > 0:
            raise ValueError("The elastic slip must be positive.")

        self.elastic_slip = elastic_slip

        curvatures = ["converged", "trial", None]
        if curvature not in curvatures:
            raise ValueError(f"The curvature must be one of {curvatures}.")

        self.curvature = curvature

        # the converged state variables, for which the multipliers were evaluated
        self._multipliers_statevars = None

        self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
        self.results = Results(stress=False, elasticity=False)

        # differences of undeformed coordinates between points in potential contact
        # and center-point (initial gap vectors)
        self.dX = self.mesh.points[self.points] - self.mesh.points[self.centerpoint]

        # initial state variables: gap vectors, reference (stick) gap vectors and
        # contact flags of the last converged increment
        dx = self._gap_vectors()
        active = self._kinematics(dx)[0] < 0.0

        self.results.statevars = {"dx": dx, "dx_ref": dx.copy(), "active": active}
        self.results._statevars = None

        # trial results of the last evaluation
        self.results.dx_ref = dx.copy()
        self.results.active = active.copy()
        self.results.slip = np.zeros(len(self.points), dtype=bool)

    @abstractmethod
    def _kinematics(self, dx):
        """Return the gaps, the outward normal vectors and their derivatives w.r.t.
        the gap vectors (or None for constant normal vectors)."""

    def _gap_vectors(self):
        "Return the gap vectors between the points and the center-point."

        u = self.field[0].values

        return self.dX + (u[self.points] - u[self.centerpoint])

    def checkpoint(self):
        """Return a checkpoint of the state of the contact.

        Returns
        -------
        dict
            A dict with copies of the state variables of the last converged increment
            (the gap vectors, the reference gap vectors of the tangential (frictional)
            contact and the mask of active contact points) and the multipliers.

        Notes
        -----
        The state variables are only updated after a converged increment. The field is
        not included in the checkpoint, see e.g.
        :meth:`SolidBody.checkpoint() <felupe.SolidBody.checkpoint>`.

        See Also
        --------
        felupe.CutbackPlugin : A cutback of the increment of failed substeps.
        """

        statevars = {k: v.copy() for k, v in self.results.statevars.items()}

        return {"results.statevars": statevars, "multipliers": self.multipliers}

    def restore(self, checkpoint):
        """Restore a checkpoint of the state of the contact inplace.

        Parameters
        ----------
        checkpoint : dict
            A dict with checkpoint arrays.
        """

        statevars = {k: v.copy() for k, v in checkpoint["results.statevars"].items()}

        self.results.statevars = statevars
        self.results._statevars = None

        self.results.dx_ref = statevars["dx_ref"].copy()
        self.results.active = statevars["active"].copy()
        self.results.slip = np.zeros(len(self.points), dtype=bool)

        # multipliers which are updated per increment are re-evaluated
        self.multipliers = checkpoint["multipliers"]
        self._multipliers_statevars = None

        # reset force and stiffness
        self.results.force = None
        self.results.stiffness = None

    def contact_multipliers(self):
        """Return the normal and tangential multipliers.

        Returns
        -------
        ContactMultipliers
            The normal and the tangential multipliers. If items are given, the scale
            factors are multiplied by the mean normal and the mean tangential stiffness
            of the diagonal blocks of the stiffness matrices of the items, evaluated at
            all points in potential contact.
        """

        base_multiplier = 1.0
        base_multiplier_tangential = 1.0

        npoints = len(self.points)

        if self.items is not None and npoints > 0:
            dim = self.mesh.dim
            dof = self.field[0].indices.dof[self.points]

            shape = (npoints, dim, dim)
            rows = np.broadcast_to(dof[:, :, None], shape).ravel()
            cols = np.broadcast_to(dof[:, None, :], shape).ravel()

            blocks = np.zeros(shape)

            for item in self.items:
                if item.results.stiffness is None:
                    _ = item.assemble.matrix(self.field)

                values = item.results.stiffness[rows, cols]
                blocks += np.asarray(values).reshape(shape)

            # normal and tangential parts of the diagonal blocks
            normal = self._kinematics(self._gap_vectors())[1]
            stiffness_normal = np.einsum("aij,ai,aj->", blocks, normal, normal)
            stiffness_tangential = np.einsum("aii->", blocks) - stiffness_normal

            base_multiplier = stiffness_normal / npoints
            base_multiplier_tangential = stiffness_tangential / (
                npoints * max(dim - 1, 1)
            )

        return ContactMultipliers(
            self.multiplier * base_multiplier,
            self.multiplier_tangential * base_multiplier_tangential,
        )

    def _update_multipliers(self):
        "Evaluate the multipliers, depending on the multiplier update."

        statevars = self.results.statevars

        update = self.multipliers is None or self.multiplier_update == "iteration"

        if self.multiplier_update == "increment":
            update = update or (self._multipliers_statevars is not statevars)

        if update:
            self.multipliers = self.contact_multipliers()
            self._multipliers_statevars = statevars

        return self.multipliers

    @staticmethod
    def _tangential(w, n, projection, curvature):
        """Return the projections of vectors to the tangent planes and their
        derivatives w.r.t. the gap vectors (the vectors depend on the gap vectors
        with an identity derivative)."""

        e = np.einsum("aij,aj->ai", projection, w)
        E = projection.copy()

        if curvature is not None:
            nw = np.einsum("ai,ai->a", n, w)
            Hw = np.einsum("aij,aj->ai", curvature, w)
            E -= nw.reshape(-1, 1, 1) * curvature + np.einsum("ai,aj->aij", n, Hw)

        return e, E

    def _evaluate(self, field=None):
        """Evaluate the contact forces and the stiffness blocks of the points in
        contact for the trial state, based on the converged state variables."""

        if field is not None:
            self.field = field

        dim = self.mesh.dim
        statevars = self.results.statevars

        dx = self._gap_vectors()
        dtype = dx.dtype

        gap, normal, curvature = self._kinematics(dx)
        active = gap < 0.0
        contact = np.where(active)[0]
        m = len(contact)

        # trial reference gap vectors and sliding flags
        dx_ref = dx.copy()
        slip = np.zeros(len(self.points), dtype=bool)

        force = np.zeros((m, dim), dtype=dtype)
        stiffness = np.zeros((m, dim, dim), dtype=dtype)

        # multipliers (evaluated on the first evaluation within an increment)
        multipliers = self._update_multipliers()
        lambda_n = multipliers.normal
        lambda_t = multipliers.tangential

        if m > 0:
            g = gap[contact]
            n = normal[contact]
            H = None if curvature is None else curvature[contact]
            nn = np.einsum("ai,aj->aij", n, n)

            # normal contribution
            force += lambda_n * g.reshape(-1, 1) * n
            stiffness += lambda_n * nn

            if H is not None and self.curvature is not None:
                # curvature term of the normal stiffness, evaluated with the gaps of
                # the trial state (consistent) or with the gaps of the last converged
                # increment (zero for points which come into contact)
                g_curvature = g

                if self.curvature == "converged":
                    g_n = self._kinematics(statevars["dx"][contact])[0]
                    g_curvature = np.where(
                        statevars["active"][contact], np.minimum(g_n, 0.0), 0.0
                    )

                stiffness += lambda_n * g_curvature.reshape(-1, 1, 1) * H

            if self.friction > 0.0:
                dx_c = dx[contact]
                P_t = np.eye(dim) - nn

                # trial elastic slip and its derivative w.r.t. the gap vector
                e, E = self._tangential(dx_c - statevars["dx_ref"][contact], n, P_t, H)

                # points which come into contact within the increment: the reference
                # is the interpolated point of first contact (zero gap)
                new = ~statevars["active"][contact]

                if np.any(new):
                    dx_n = statevars["dx"][contact][new]
                    g_n = self._kinematics(dx_n)[0]
                    g_new = g[new]

                    delta, D = self._tangential(
                        dx_c[new] - dx_n,
                        n[new],
                        P_t[new],
                        None if H is None else H[new],
                    )

                    s = np.ones_like(g_new)
                    ds = np.zeros_like(g_new)

                    first = g_n > 0.0
                    s[first] = -g_new[first] / (g_n[first] - g_new[first])
                    ds[first] = -g_n[first] / (g_n[first] - g_new[first]) ** 2

                    e[new] = s.reshape(-1, 1) * delta
                    E[new] = s.reshape(-1, 1, 1) * D + ds.reshape(-1, 1, 1) * np.einsum(
                        "ai,aj->aij", delta, n[new]
                    )

                force_n_abs = -lambda_n * g
                force_t_limit = self.friction * force_n_abs

                # tangential multipliers: constant or proportional to the normal force
                if self.elastic_slip is None:
                    lambda_t_points = np.full(m, lambda_t, dtype=dtype)
                else:
                    mu = self.friction if np.isfinite(self.friction) else 1.0
                    lambda_t_points = mu * force_n_abs / self.elastic_slip

                force_t_trial = lambda_t_points.reshape(-1, 1) * e
                force_t_norm = np.linalg.norm(force_t_trial, axis=1)

                # points with friction: sticking or sliding
                stick = np.ones(m, dtype=bool)
                if np.isfinite(self.friction):
                    tol = np.sqrt(np.finfo(float).eps) * np.maximum(force_t_limit, 1.0)
                    stick = force_t_norm <= (force_t_limit + tol)

                slide = (~stick) & (force_t_norm > 0.0)

                force_t = force_t_trial
                stiffness_t = lambda_t_points.reshape(-1, 1, 1) * E

                # elastic slip of the (trial) reference gap vectors
                e_ref = e

                if self.elastic_slip is not None:
                    # derivative of the tangential multipliers w.r.t. the gap vector
                    stiffness_t -= (mu * lambda_n / self.elastic_slip) * np.einsum(
                        "ai,aj->aij", e, n
                    )

                if np.any(slide):
                    t_hat = force_t_trial[slide] / force_t_norm[slide].reshape(-1, 1)
                    e_norm = np.linalg.norm(e[slide], axis=1)

                    force_t = force_t_trial.copy()
                    force_t[slide] = force_t_limit[slide].reshape(-1, 1) * t_hat

                    # return mapping of the elastic slip
                    e_ref = e.copy()
                    e_ref[slide] *= (
                        force_t_limit[slide] / force_t_norm[slide]
                    ).reshape(-1, 1)

                    # consistent stiffness of the return mapping
                    projection_slip = (
                        np.eye(dim) - np.einsum("ai,aj->aij", t_hat, t_hat)
                    ) / e_norm.reshape(-1, 1, 1)

                    stiffness_t[slide] = self.friction * (
                        force_n_abs[slide].reshape(-1, 1, 1)
                        * projection_slip
                        @ E[slide]
                        - lambda_n * np.einsum("ai,aj->aij", t_hat, n[slide])
                    )

                force += force_t
                stiffness += stiffness_t

                slip[contact] = slide

                # updated (trial) reference gap vectors
                dx_ref[contact] = dx_c - e_ref

        if self.symmetric:
            stiffness = (stiffness + stiffness.transpose(0, 2, 1)) / 2

        # trial state variables, these are taken after a converged increment
        self.results._statevars = {"dx": dx, "dx_ref": dx_ref, "active": active}

        self.results.dx_ref = dx_ref
        self.results.active = active
        self.results.slip = slip

        return ContactTrial(contact, force, stiffness, dx, dx_ref, active, slip)

    def _vector(self, field=None, parallel=False):

        trial = self._evaluate(field)
        dof = self.field[0].indices.dof

        self.results.force = _assemble_vector(
            dof=dof[self.points[trial.contact]],
            dof_center=dof[self.centerpoint],
            forces=trial.force,
            size=self.mesh.npoints * self.mesh.dim,
            dtype=trial.dx.dtype,
        )

        return self.results.force

    def _matrix(self, field=None, parallel=False):

        trial = self._evaluate(field)
        dof = self.field[0].indices.dof

        self.results.stiffness = _assemble_matrix(
            dof=dof[self.points[trial.contact]],
            dof_center=dof[self.centerpoint],
            blocks=trial.stiffness,
            size=self.mesh.npoints * self.mesh.dim,
            dtype=trial.dx.dtype,
        )

        return self.results.stiffness

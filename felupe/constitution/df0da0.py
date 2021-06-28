# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from types import SimpleNamespace

from ..math import (
    dot,
    ddot,
    ddot44,
    ddot444,
    transpose,
    majortranspose,
    inv,
    dya,
    cdya,
    cdya_ik,
    cdya_il,
    det,
    eigh,
    identity,
    trace,
    dev,
)


class Composite:
    def __init__(self, *args):

        self.materials = args
        self.kind = SimpleNamespace(**{"df": 0, "da": 0})

    def stress(self, *args, **kwargs):
        return np.sum([m.stress(*args) for m in self.materials], 0)

    def elasticity(self, *args, **kwargs):
        return np.sum([m.elasticity(*args, **kwargs) for m in self.materials], 0)


class Material:
    def __init__(self, stress, elasticity):
        """Total-Lagrange Material class:
        stress = 2nd Piola-Kirchhoff stress S
        elasticity = associated 4th-order elasticity tensor C4 = dS/dE
        """

        self.stress = stress
        self.elasticity = elasticity
        self.kind = SimpleNamespace(**{"df": 0, "da": 0})


class InvariantBased:
    def __init__(self, umat, tol=np.finfo(float).eps):
        self.umat = umat
        self.kind = SimpleNamespace(**{"df": 0, "da": 0})
        self.C = 0

    def update(self, C):

        if np.all(C == self.C):
            pass
        else:
            self.invariants = np.array(
                [trace(C), (trace(C) ** 2 - trace(dot(C, C))) / 2, det(C)]
            )
            self.W_a, self.W_ab = self.umat(self.invariants)

    def stress(self, C):
        self.update(C)

        S = np.zeros_like(C)

        I1, I2, I3 = self.invariants
        W1, W2, W3 = self.W_a

        if not np.all(W1 == 0):
            eye = identity(C)
            S += 2 * W1 * eye

        if not np.all(W2 == 0):
            eye = identity(C)
            S += 2 * W2 * (I1 * eye - C)

        if not np.all(W3 == 0):
            S += 2 * W3 * I3 * inv(C)

        return S

    def elasticity(self, C):
        self.update(C)

        ndim, ngauss, nelems = C.shape[-3:]
        C4 = np.zeros((ndim, ndim, ndim, ndim, ngauss, nelems))

        I = identity(C)

        I1, I2, I3 = self.invariants
        W1, W2, W3 = self.W_a
        W11 = self.W_ab[0, 0]
        W22 = self.W_ab[1, 1]
        W33 = self.W_ab[2, 2]
        W12 = self.W_ab[0, 1]
        W13 = self.W_ab[0, 2]
        W23 = self.W_ab[1, 2]

        a00 = W11 + W2 + I1 * W12
        a11 = W22
        a22 = W33 * I3 ** 2 + W3 * I3
        a01 = -(W12 + I1 * W22)
        a02 = W13 * I3 + W23 * I3 * I1
        a12 = -W23 * I3
        b00 = -W2
        b22 = W3 * I3

        if not np.all(a00 == 0):
            C4 += 4 * a00 * dya(I, I)

        if not np.all(a11 == 0):
            C4 += 4 * a11 * dya(C, C)

        if not np.all(a22 == 0):
            C4 += 4 * a22 * dya(inv(C), inv(C))

        if not np.all(a01 == 0):
            C4 += 4 * a01 * (dya(I, C) + dya(C, I))

        if not np.all(a02 == 0):
            C4 += 4 * a02 * (dya(I, inv(C)) + dya(inv(C), I))

        if not np.all(a12 == 0):
            C4 += 4 * a12 * (dya(C, inv(C)) + dya(inv(C), C))

        if not np.all(b00 == 0):
            C4 += 4 * b00 * cdya(I, I)

        if not np.all(b22 == 0):
            C4 += 4 * b22 * cdya(inv(C), inv(C))

        return C4


class PrincipalStretchBased:
    def __init__(self, umat, tol=np.finfo(float).eps):
        self.umat = umat
        self.kind = SimpleNamespace(**{"df": 0, "da": 0})
        self.C = 0
        self.tol = tol

    def update(self, C):
        if np.all(C == self.C):
            pass
        else:
            wC, vC = eigh(C)

            self.stretches = np.sqrt(wC)
            self.bases = np.array([dya(N, N, mode=1) for N in transpose(vC)])

            self.W_a, self.W_ab = self.umat(self.stretches)

    def stress(self, C):
        self.update(C)
        self.stresses = self.W_a / self.stretches
        return np.sum([Sa * Ma for Sa, Ma in zip(self.stresses, self.bases)], 0)

    def elasticity(self, C):
        self.update(C)

        ndim, ngauss, nelems = C.shape[-3:]
        C4 = np.zeros((ndim, ndim, ndim, ndim, ngauss, nelems))

        for a in range(3):
            Wa = self.W_a[a]
            Ma = self.bases[a]
            la = self.stretches[a]

            C4 -= Wa / la ** 3 * dya(Ma, Ma)

            for b in range(3):
                Wb = self.W_a[b]
                Wab = self.W_ab[a, b]
                Mb = self.bases[b]
                lb = self.stretches[b]

                C4 += Wab / la / lb * dya(Ma, Mb)

                if b != a:
                    la[abs(la - lb) < self.tol] += self.tol
                    Gab = cdya(Ma, Mb) + cdya(Mb, Ma)
                    C4 += (Wa / la - Wb / lb) / (la ** 2 - lb ** 2) * Gab

        return C4


class StrainInvariantBased(PrincipalStretchBased):
    def __init__(
        self,
        umat_straininvariants,
        strain=lambda stretch, k: 1 / k * (stretch ** k - 1),
        dstraindstretch=lambda stretch, k: stretch ** (k - 1),
        d2straindstretch2=lambda stretch, k: (k - 1) * stretch ** (k - 2),
        k=2,
        tol=np.finfo(float).eps,
    ):

        self.umat_straininvariants = umat_straininvariants
        self.strain = strain
        self.dstraindstretch = dstraindstretch
        self.d2straindstretch2 = d2straindstretch2
        self.k = k

        super().__init__(self.umat_stretch, tol=tol)

    def umat_stretch(self, stretches):
        E_k = self.strain(stretches, self.k)

        I1 = np.sum(E_k, 0)
        I2 = np.sum(E_k ** 2, 0)

        straininvariants = I1, I2

        w_i, w_ij = self.umat_straininvariants(straininvariants)

        dEdl = self.dstraindstretch(stretches, self.k)
        d2Edl2 = self.d2straindstretch2(stretches, self.k)

        dIdE = np.array([np.ones_like(E_k), 2 * E_k])
        d2IdE2 = np.array([np.zeros_like(E_k), 2 * np.ones_like(E_k)])

        ndim, ngauss, nelems = stretches.shape

        W_a = np.zeros((ndim, ngauss, nelems))
        W_ab = np.zeros((ndim, ndim, ngauss, nelems))

        for i in range(2):

            for a in range(ndim):
                W_a[a] += w_i[i] * dIdE[i, a] * dEdl[a]
                W_ab[a, a] += (
                    w_i[i] * dIdE[i, a] * d2Edl2[a] + w_i[i] * dEdl[a] * d2IdE2[i, a]
                )

                for j in range(2):
                    for b in range(ndim):
                        W_ab[a, b] += (
                            dEdl[a] * dIdE[i, a] * w_ij[i, j] * dIdE[j, b] * dEdl[b]
                        )

        return W_a, W_ab


class Hydrostatic:
    def __init__(self, bulk):
        self.bulk = bulk
        self.kind = SimpleNamespace(**{"df": 0, "da": 0})

    def dUdJ(self, J):
        return self.bulk * (J - 1)

    def d2UdJdJ(self, J):
        return self.bulk

    def stress(self, F, J, C, invC):
        return self.dUdJ(J) * J * invC

    def elasticity(self, F, J, C, invC):
        p = self.dUdJ(J)
        q = p + self.d2UdJdJ(J) * J
        return J * (q * dya(invC, invC) - 2 * p * cdya(invC, invC))


class AsIsochoric:
    def __init__(self, material_isochoric):
        self.isochoric = material_isochoric
        self.kind = SimpleNamespace(**{"df": 0, "da": 0})

    def stress(self, F, J, C, invC):
        Cu = J ** (-2 / 3) * C
        Sb = J ** (-2 / 3) * self.isochoric.stress(Cu)
        return Sb - ddot(Sb, C) / 3 * invC

    def elasticity(self, F, J, C, invC):
        eye = identity(C)
        P4 = cdya(eye, eye) - dya(invC, C) / 3

        Cu = J ** (-2 / 3) * C
        Sb = J ** (-2 / 3) * self.isochoric.stress(Cu)

        C4u = self.isochoric.elasticity(Cu)
        if np.all(C4u == 0):
            PC4bP = C4u
        else:
            C4b = J ** (-4 / 3) * C4u
            PC4bP = ddot444(P4, C4b, majortranspose(P4))

        SbC = ddot(Sb, C)

        return (
            PC4bP
            - 2 / 3 * (dya(Sb, invC) + dya(invC, Sb))
            + 2 / 9 * SbC * dya(invC, invC)
            + 2 / 3 * SbC * cdya(invC, invC)
        )


class AsFull:
    def __init__(self, material):
        self.material = material
        self.kind = SimpleNamespace(**{"df": 0, "da": 0})

    def stress(self, F, J, C, invC):
        return self.material.stress(C)

    def elasticity(self, F, J, C, invC):
        return self.material.elasticity(C)

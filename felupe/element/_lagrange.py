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
from scipy.special import factorial
from string import ascii_lowercase as alphabet
from copy import deepcopy as copy


class ArbitraryOrderLagrange:
    "Lagrange quad/hexahdron finite element of arbitrary order."

    def __init__(self, order, ndim, interval=(-1, 1)):
        self.ndim = ndim
        self.order = order
        self.npoints = (order + 1) ** ndim
        self.nbasis = self.npoints
        self.interval = interval

        self._nbasis = order + 1

        # init curve-parameter matrix
        n = self._nbasis
        self._AT = np.linalg.inv(
            np.array([self._polynomial(p, n) for p in self._points(n)])
        ).T

        # indices for outer product in einstein notation
        # idx = ["a", "b", "c", ...][:dim]
        # subscripts = "a,b,c -> abc"
        self._idx = [letter for letter in alphabet][: self.ndim]
        self._subscripts = ",".join(self._idx) + "->" + "".join(self._idx)

    def basis(self, r):
        "Basis function vector at coordinate vector r."
        n = self._nbasis

        # 1d - basis function vectors per axis
        h = [self._AT @ self._polynomial(ra, n) for ra in r]

        return np.einsum(self._subscripts, *h).ravel("F")

    def basisprime(self, r):
        "Basis function derivative vector at coordinate vector r."
        n = self._nbasis

        # 1d - basis function vectors per axis
        h = [self._AT @ self._polynomial(ra, n) for ra in r]

        # shifted 1d - basis function vectors per axis
        k = [self._AT @ np.append(0, self._polynomial(ra, n)[:-1]) for ra in r]

        # init output
        dhdr = np.zeros((n ** self.ndim, self.ndim))

        # loop over columns
        for i in range(self.ndim):
            g = copy(h)
            g[i] = k[i]
            dhdr[:, i] = np.einsum(self._subscripts, *g).ravel("F")

        return dhdr

    def _points(self, n):
        "Equidistant n points in interval [-1, 1]."
        return np.linspace(*self.interval, n)

    def _polynomial(self, r, n):
        "Lagrange-Polynomial of order n evaluated at coordinate r."
        m = np.arange(n)
        return r ** m / factorial(m)

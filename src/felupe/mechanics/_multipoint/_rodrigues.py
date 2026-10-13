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

from math import factorial

import numpy as np
from numpy.polynomial import Polynomial


def _skew(w):
    "Return the skew-symmetric matrix [w]× of a vector with three components."
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


# skew-symmetric matrices [e_a]× of the base vectors, shape (3, 3, 3)
_SKEW_BASE = np.array([_skew(e) for e in np.eye(3)])

# power series of the coefficients alpha(s) = sin(t) / t and beta(s) = (1 - cos(t)) / s
# of the rotation matrix with s = t², along with their first and second derivatives
_POWER_SERIES = [
    Polynomial([(-1) ** k / factorial(2 * k + offset) for k in range(12)])
    for offset in [1, 2]
]
_SERIES = [[series.deriv(m) for m in range(3)] for series in _POWER_SERIES]


def _rotation_coefficients(s):
    r"""Return the coefficients :math:`\alpha(s) = \sin(t) / t` and
    :math:`\beta(s) = (1 - \cos(t)) / t^2` of the rotation matrix with :math:`s = t^2`,
    along with their first and second derivatives w.r.t. :math:`s`.

    Notes
    -----
    The closed-form expressions are undefined for :math:`s = 0`, which is the state at
    the beginning of each increment of the rotation. Further, their derivatives are
    differences of nearly equal numbers, divided by :math:`s` (or :math:`s^2`), i.e.
    they suffer from cancellation for small angles, e.g. a relative error of about
    1e-2 for :math:`\beta''` at :math:`s = 10^{-4}`. Hence, the Taylor series of the
    sine and the cosine (divided by :math:`t` and :math:`t^2`, with 12 terms) are used
    for :math:`s < 1`, which are accurate to machine precision there. Both branches
    agree within 1e-13 at :math:`s = 1`."""

    if s < 1.0:
        alpha, beta = [[term(s) for term in series] for series in _SERIES]

    else:
        t = np.sqrt(s)
        cos = np.cos(t)
        a = np.sin(t) / t
        b = (1 - cos) / s
        da = (cos - a) / (2 * s)
        db = (a - 2 * b) / (2 * s)
        alpha = [a, da, -(a + 6 * da) / (4 * s)]
        beta = [b, db, (da - 4 * db) / (2 * s)]

    return alpha, beta


def _rotation(w, derivatives=True):
    r"""Return the rotation matrix
    :math:`\boldsymbol{Q} = \exp([\boldsymbol{w}]_\times)` of a rotation vector
    :math:`\boldsymbol{w}` and (optionally) its first and second derivatives w.r.t. the
    rotation vector with shapes (3, 3), (3, 3, 3) and (3, 3, 3, 3), where the axes of
    the rotation vector come first.

    Notes
    -----
    The rotation matrix is given by Rodrigues' formula with the skew-symmetric matrix
    :math:`\boldsymbol{S} = [\boldsymbol{w}]_\times`, the coefficients
    :math:`\alpha(s)` and :math:`\beta(s)` and :math:`s = \boldsymbol{w} \cdot
    \boldsymbol{w}`.

    ..  math::

        \boldsymbol{Q} = \boldsymbol{1} + \alpha\ \boldsymbol{S}
            + \beta\ \boldsymbol{S}^2

    With :math:`\boldsymbol{E}_a = [\boldsymbol{e}_a]_\times`, the derivatives are

    ..  math::

        \frac{\partial \boldsymbol{Q}}{\partial w_a} &= 2 w_a \left(
            \alpha'\ \boldsymbol{S} + \beta'\ \boldsymbol{S}^2 \right)
            + \alpha\ \boldsymbol{E}_a
            + \beta \left( \boldsymbol{E}_a \boldsymbol{S}
            + \boldsymbol{S} \boldsymbol{E}_a \right)

        \frac{\partial^2 \boldsymbol{Q}}{\partial w_a\ \partial w_b} &=
            2 \delta_{ab} \left( \alpha'\ \boldsymbol{S}
            + \beta'\ \boldsymbol{S}^2 \right)
            + 4 w_a w_b \left( \alpha''\ \boldsymbol{S}
            + \beta''\ \boldsymbol{S}^2 \right)
            + 2 w_a\ \boldsymbol{T}_b + 2 w_b\ \boldsymbol{T}_a
            + \beta \left( \boldsymbol{E}_a \boldsymbol{E}_b
            + \boldsymbol{E}_b \boldsymbol{E}_a \right)

        \boldsymbol{T}_a &= \alpha'\ \boldsymbol{E}_a
            + \beta' \left( \boldsymbol{E}_a \boldsymbol{S}
            + \boldsymbol{S} \boldsymbol{E}_a \right)

    where :math:`(\bullet)'` denotes the derivative w.r.t. :math:`s`.
    """

    w = np.asarray(w, dtype=float)
    (a, da, dda), (b, db, ddb) = _rotation_coefficients(w @ w)

    S = _skew(w)
    SS = S @ S
    Q = np.eye(3) + a * S + b * SS

    if not derivatives:
        return Q

    E = _SKEW_BASE
    ES = E @ S + S @ E
    P = da * S + db * SS
    T = da * E + db * ES
    EE = E[:, None] @ E[None, :]

    dQ = 2 * w.reshape(3, 1, 1) * P + a * E + b * ES
    ddQ = (
        2 * np.eye(3).reshape(3, 3, 1, 1) * P
        + 4 * np.outer(w, w).reshape(3, 3, 1, 1) * (dda * S + ddb * SS)
        + 2 * w.reshape(3, 1, 1, 1) * T[None, :]
        + 2 * w.reshape(1, 3, 1, 1) * T[:, None]
        + b * (EE + EE.transpose([1, 0, 2, 3]))
    )

    return Q, dQ, ddQ

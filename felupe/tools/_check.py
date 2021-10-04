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

from ..math import norm


def check(dfields, fields, f, dof1, dof0, tol_f=1e-3, tol_x=1e-3, verbose=1):
    "Check if solution dfields is valid."

    if "mixed" in str(type(fields)):
        return _check_mixed(dfields, fields, f, dof1, dof0, tol_f, tol_x, verbose)

    else:
        x = fields
        dx = dfields
        return _check_single(dx, x, f, dof1, dof0, tol_f, tol_x, verbose)


def _check_single(dx, x, f, dof1, dof0, tol_f=1e-3, tol_x=1e-3, verbose=1):
    "Check if solution dx is valid."

    # get reference values of "f" and "x"
    ref_f = 1 if np.linalg.norm(f[dof0]) == 0 else np.linalg.norm(f[dof0])
    ref_x = 1 if np.linalg.norm(x[dof0]) == 0 else np.linalg.norm(x[dof0])

    norm_f = np.linalg.norm(f[dof1]) / ref_f
    norm_x = np.linalg.norm(dx.ravel()[dof1]) / ref_x

    if verbose:
        info_r = f"|r|={norm_f:1.3e} |u|={norm_x:1.3e}"
        print(info_r)

    if norm_f < tol_f and norm_x < tol_x:
        success = True
    else:
        success = False

    return success


def _check_mixed(dfields, fields, f, dof1, dof0, tol_f=1e-3, tol_x=1e-3, verbose=1):
    "Check if solution dfields is valid."

    if "mixed" in str(type(fields)):
        fields = fields.fields

    x = fields[0]
    dx = dfields[0]

    # get reference values of "f" and "x"
    ref_f = 1 if np.linalg.norm(f[dof0]) == 0 else np.linalg.norm(f[dof0])
    ref_x = 1 if np.linalg.norm(x[dof0]) == 0 else np.linalg.norm(x[dof0])

    norm_f = np.linalg.norm(f[dof1[dof1 < len(dx)]]) / ref_f
    norm_x = np.linalg.norm(dx.ravel()[dof1[dof1 < len(dx)]]) / ref_x

    norm_dfields = norm(dfields[1:])

    if verbose:
        info_r = f"|r|={norm_f:1.3e} |u|={norm_x:1.3e}"
        info_f = [f"(|δ{2+i}|={norm_f:1.3e})" for i, norm_f in enumerate(norm_dfields)]

        print(" ".join([info_r, *info_f]))

    if norm_f < tol_f and norm_x < tol_x:
        success = True
    else:
        success = False

    return success

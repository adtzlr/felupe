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

from .helpers import ddot, transpose, inv, dya, cdya_ik, cdya_il, det, identity

class NeoHooke:
    
    def __init__(self):
        self.P = self._P_nh
        self.A = self._A_nh

    def _P_nh(self, F, p, J, *args):
        mu = args[0]
        iFT = transpose(inv(F))
        
        Pdev = mu * (F - ddot(F,F)/3*iFT) * J**(-2/3)
        Pvol = p * J * iFT
        
        return Pdev + Pvol
    
    def _A_nh(self, F, p, J, *args):
        mu = args[0]
        
        detF = det(F)
        iFT = transpose(inv(F))
        eye = identity(F)
    
        A4_dev = mu * (cdya_ik(eye, eye)
                       - 2/3 * dya(F, iFT)
                       - 2/3 * dya(iFT, F)
                       + 2/9 * ddot(F, F) * dya(iFT, iFT)
                       + 1/3 * ddot(F, F) * cdya_il(iFT, iFT)
                       ) * detF**(-2/3)
    
        A4_vol = p * J * (dya(iFT,iFT) - cdya_il(iFT,iFT)) \
    
        return A4_dev + A4_vol
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

from multiprocessing import cpu_count


class MatadiMaterial:
    def __init__(self, material, threads=cpu_count()):
        "Wrap a MatADi Materials into a FElupe material."
        self.material = material
        self.threads = threads

    def function(self, *args):
        fun = self.material.function(args, threads=self.threads)
        if len(fun) == 1:
            return fun[0]
        else:
            return fun

    def gradient(self, *args):
        grad = self.material.gradient(args, threads=self.threads)
        if len(grad) == 1:
            return grad[0]
        else:
            return grad

    def hessian(self, *args):
        hess = self.material.hessian(args, threads=self.threads)
        if len(hess) == 1:
            return hess[0]
        else:
            return hess

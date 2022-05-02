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


class Assemble:
    "A class with assembly methods of a SolidBody."

    def __init__(self, vector, matrix):
        self.vector = vector
        self.matrix = matrix


class Evaluate:
    "A class with evaluate methods of a SolidBody."

    def __init__(self, gradient, hessian, cauchy_stress=None):
        self.gradient = gradient
        self.hessian = hessian

        if cauchy_stress is not None:
            self.cauchy_stress = cauchy_stress


class Results:
    "A class with intermediate results of a SolidBody."

    def __init__(self, stress=False, elasticity=False):

        self.force = None
        self.stiffness = None
        self.kinematics = None

        if stress:
            self.stress = None

        if elasticity:
            self.elasticity = None

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
    """Wrap a matADi Materials into a FElupe material. While matADi always
    requires and returns a list of input parameters / return values, FElupe
    takes one or more arguments and returns an array or a list of arrays.

    Arguments
    ---------
    material : matadi.MaterialHyperelastic
        Hyperelastic material from matADi
    threads : int, optional
        Number of threads (default is number of CPU cores)

    """

    def __init__(self, material, threads=cpu_count()):

        self.material = material
        self.threads = threads

    def function(self, *args):
        """Evaluate the function.

        Arguments
        ---------
        *args : tuple
            Input arguments

        Returns
        -------
        list or ndarray
            The evaluated function either as array if one input argument is
            passed or as list of arrays if more than one input arguments are
            provided.

        """
        fun = self.material.function(args, threads=self.threads)
        if len(fun) == 1:
            return fun[0]
        else:
            return fun

    def gradient(self, *args):
        """Evaluate the gradient.

        Arguments
        ---------
        *args : tuple
            Input arguments

        Returns
        -------
        list or ndarray
            The evaluated gradient either as array if one input argument is
            passed or as list of arrays if more than one input arguments are
            provided.

        """
        grad = self.material.gradient(args, threads=self.threads)
        if len(grad) == 1:
            return grad[0]
        else:
            return grad

    def hessian(self, *args):
        """Evaluate the hessian.

        Arguments
        ---------
        *args : tuple
            Input arguments

        Returns
        -------
        list or ndarray
            The evaluated hessian either as array if one input argument is
            passed or as list of arrays if more than one input arguments are
            provided.

        """
        hess = self.material.hessian(args, threads=self.threads)
        if len(hess) == 1:
            return hess[0]
        else:
            return hess

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


from .. import __version__ as version


class Job:
    "A job with a list of steps."

    def __init__(self, steps, callback=lambda stepnumber, substepnumber, substep: None):
        self.steps = steps
        self.nsteps = len(steps)
        self.callback = callback

    def evaluate(self, verbose=True, **kwargs):

        if verbose:
            print(
                f"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|
FElupe Version {version}

"""
            )

            print("Run Job")
            print("=======\n")

        for j, step in enumerate(self.steps):

            print(f"Begin Evaluation of Step {j + 1}.")

            substeps = step.generate(verbose=verbose, **kwargs)
            for i, substep in enumerate(substeps):

                if verbose:
                    _substep = f"Substep {i}/{step.nsubsteps - 1}"
                    _step = f"Step {j + 1}/{self.nsteps}"

                    print(f"{_substep} of {_step} successful.")

                self.callback(j, i, substep)

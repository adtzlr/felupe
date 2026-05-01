# -*- coding: utf-8 -*-
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

import os
import numpy as np
from time import perf_counter

from ._misc import logo, runs_on


def print_header():
    print("\n".join([logo(), runs_on(), "", "Run Job", "======="]))


class ProgressPlugin:
    r"""A job plugin to output the progress of a job evaluation in the terminal.

    Parameters
    ----------
    verbose : bool or int or None, optional
        Verbosity level to control how messages are printed during evaluation. If
        1 or True and ``tqdm`` is installed, a progress bar is shown. If ``tqdm`` is
        missing or verbose is 2, more detailed text-based messages are printed.
        Default is None. If None, verbosity is set to True. If None and the
        environmental variable FELUPE_VERBOSE is set and its value is not ``true``,
        then logging is turned off.
    tqdm : str, optional
        If verbose is True, choose a backend for ``tqdm`` ("tqdm", ``"auto"`` or
        ``"notebook"``. Default is ``"tqdm"``.

    Notes
    -----
    Requires ``tqdm`` for an interactive progress bar if ``verbose=True``.

    See Also
    --------
    Job : A job with a list of steps and a method to evaluate them.
    """

    def __init__(
        self,
        verbose=None,
        tqdm="tqdm",
    ):
        self.configure(verbose=verbose, tqdm=tqdm)

    def configure(self, verbose, tqdm):
        self.verbose = verbose
        self.tqdm = tqdm

    def before_job(self, context, state):

        if self.verbose is None:
            FELUPE_VERBOSE = os.environ.get("FELUPE_VERBOSE")
            if FELUPE_VERBOSE is None:
                self.verbose = True
            else:
                self.verbose = FELUPE_VERBOSE == "true"

        if self.verbose:
            try:
                backend = str(self.tqdm).lower()

                if backend == "tqdm":
                    from tqdm import tqdm
                elif backend == "auto":
                    from tqdm.auto import tqdm
                elif backend == "notebook":
                    from tqdm.notebook import tqdm
                else:
                    raise ValueError('tqdm must be "tqdm", "auto" or "notebook".')

                self._tqdm = tqdm

            except ModuleNotFoundError:  # pragma: no cover
                self.verbose = 2  # pragma: no cover

        if self.verbose == 2:
            print_header()

        if self.verbose == 1:
            total = sum([step.nsubsteps for step in context.job.steps])
            self.progress_bar = self._tqdm(
                total=total,
                desc="Step   ",
                unit="substep",
                colour="green",
            )
            self.progress_bar_newton = self._tqdm(
                total=100,
                desc="Substep",
                colour="cyan",
                unit="%",
            )
        else:
            self.progress_bar_newton = None

    def before_step(self, context, state):
        if self.verbose == 2:
            print(f"Begin Evaluation of Step {state.stepnumber + 1}.")

    def before_newton(self, context, state):
        if self.verbose:
            self.decades = None

        if self.verbose == 1:
            if self.progress_bar_newton is None:
                self.progress_bar_newton = self._tqdm(
                    total=100, colour="yellow", unit="%"
                )
                self.close_bar = True
            else:
                self.progress_bar_newton.reset()
                self.close_bar = False

            self.progress0 = 0
            self.progress = 0

        if self.verbose == 2:
            self.runtimes = [perf_counter()]
            self.soltimes = []

            print()
            print("Newton-Raphson solver")
            print("=====================")
            print()
            print("| # | norm(fun) |  norm(dx) |")
            print("|---|-----------|-----------|")

    def before_linear_solve(self, context, state):
        if self.verbose == 2:
            self.soltime_start = perf_counter()

    def after_linear_solve(self, context, state):
        if self.verbose == 2:
            self.soltime_end = perf_counter()
            self.soltimes.append([self.soltime_start, self.soltime_end])

    def after_iteration(self, context, state):
        # update progress bar if norm of residuals is available
        if self.verbose == 1:
            completion = 0.1

            if state.fnorm > 0.0:
                # initial log. ratio of first residual norm vs. tolerance norm
                if self.decades is None:
                    self.decades = max(1.0, np.log10(state.fnorm) - np.log10(state.tol))

                # current log. ratio of first residual norm vs. tolerance norm
                dfnorm = np.log10(state.fnorm) - np.log10(state.tol)
                completion += 0.9 * (1 - dfnorm / self.decades)

            # progress in percent, ensure lower equal 100%
            self.progress = np.clip(100 * completion, 0, 100).astype(int)
            self.progress_bar_newton.update(self.progress - self.progress0)
            self.progress0 = self.progress

        if self.verbose == 2:
            print(
                "|%2d | %1.3e | %1.3e |"
                % (1 + state.iteration, state.fnorm, state.xnorm)
            )

    def after_newton(self, context, state):

        if self.verbose == 1:
            self.progress_bar_newton.update(100 - self.progress)
            if self.close_bar:
                self.progress_bar_newton.close()

        if self.verbose == 2:
            self.runtimes.append(perf_counter())
            runtime = np.diff(self.runtimes)[0]
            soltime = np.diff(self.soltimes).sum()
            print(
                "\nConverged in %d iterations (Assembly: %1.4g s, Solve: %1.4g s).\n"
                % (state.iteration + 1, runtime - soltime, soltime)
            )

    def after_substep(self, context, state):
        if self.verbose == 2:
            _substep = f"Substep {state.substepnumber + 1}/{context.step.nsubsteps}"
            _step = f"Step {state.stepnumber + 1}/{len(context.job.steps)}"

            print(f"{_substep} of {_step} successful.")

        if self.verbose == 1:
            self.progress_bar.update(1)

    def after_job(self, context, state):
        if self.verbose == 1:
            self.progress_bar.close()
            self.progress_bar_newton.close()

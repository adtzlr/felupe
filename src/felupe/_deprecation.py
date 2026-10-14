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
import sys
import warnings

_PACKAGE = os.path.normcase(os.path.dirname(os.path.abspath(__file__)) + os.sep)


def _in_package(frame):
    "Return True if the code of a frame is located inside of FElupe."
    return os.path.normcase(frame.f_code.co_filename).startswith(_PACKAGE)


def warn_deprecated(message):
    """Emit a :class:`DeprecationWarning`, which is attributed to the first frame
    outside of FElupe, i.e. to the line of the code of the user.

    Python shows a :class:`DeprecationWarning` by default only if it is attributed
    to the code of ``__main__``, e.g. a script or a notebook. A fixed ``stacklevel``
    would point into FElupe if the deprecated feature is used indirectly, e.g. a
    deprecated argument of :func:`~felupe.newtonraphson` which is passed by
    :meth:`~felupe.Job.evaluate`, and the warning would be hidden.
    """

    # the frame of the caller of this function (inside FElupe)
    frame = sys._getframe(1)
    stacklevel = 2

    while frame is not None and _in_package(frame):
        frame = frame.f_back
        stacklevel += 1

    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

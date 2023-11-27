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

import numpy as np


def linsteps(points, num=10, endpoint=True, axis=None, axes=None):
    """Return a sequence from batches of evenly spaced samples between pairs of
    milestone points.

    Parameters
    ----------
    points : array_like
        The milestone values of the sequence.
    num : int, optional
        Number of samples (without the end point) to generate. Must be non-negative
        (default is 10).
    endpoint : bool, optional
        If True, ``points[-1]`` is the last sample. Otherwise, it is not included
        (default is True).
    axis : int or None, optional
        The axis in the result to store the samples. By default (None), the samples will
        be concatenated along the existing first axis. If ``axis > 0``, the samples will
        be inserted at column ``axis``. Only positive integers are supported.
    axes : int or None, optional
        Number of output columns if ``axis`` is not None. Requires ``axes > axis`` and
        will be set to ``axes = axis + 1`` by default (None).

    Returns
    -------
    samples : ndarray
        Concatenated batches of ``num`` equally spaced samples.

    Examples
    --------
    >>> import felupe as fem

    >>> fem.math.linsteps([0, 0.5, 1.5, 3.5], num=2)
    array([0.  , 0.25, 0.5 , 1.  , 1.5 , 2.5 , 3.5 ])

    Not including the end value of the sequence does not change the step size.

    >>> fem.math.linsteps([0, 0.5, 1.5, 3.5], num=2, endpoint=False)
    array([0.  , 0.25, 0.5 , 1.  , 1.5 , 2.5 ])

    >>> fem.math.linsteps([0, 0.5, 1.5, 3.5], num=2, axis=1)
    array([[0.  , 0.  ],
           [0.  , 0.25],
           [0.  , 0.5 ],
           [0.  , 1.  ],
           [0.  , 1.5 ],
           [0.  , 2.5 ],
           [0.  , 3.5 ]])

    Output with four columns:

    >>> fem.math.linsteps([0, 0.5, 1.5, 3.5], num=2, axis=1, axes=4)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.25, 0.  , 0.  ],
           [0.  , 0.5 , 0.  , 0.  ],
           [0.  , 1.  , 0.  , 0.  ],
           [0.  , 1.5 , 0.  , 0.  ],
           [0.  , 2.5 , 0.  , 0.  ],
           [0.  , 3.5 , 0.  , 0.  ]])

    The ouput degenerates to the end value

    >>> fem.math.linsteps([0, 0.5, 1.5, 3.5], num=0)
    array([3.5])

    or an empty array.

    >>> fem.math.linsteps([0, 0.5, 1.5, 3.5], num=0, endpoint=False)
    array([], dtype=float64)

    See Also
    --------
    numpy.linspace : Return evenly spaced numbers over a specified interval.
    """
    points = np.array(points).ravel()
    start = points[:-1]
    end = points[1:]
    num = np.array([num]).ravel()

    if len(num) == 1:
        num = np.tile(num, max(1, len(start)))

    num = np.pad(num, (0, max(0, len(start) - len(num))), mode="edge")

    steplist = [
        np.linspace(a, b, n, endpoint=False) for a, b, n in zip(start, end, num)
    ]
    if len(steplist) > 0:
        steps = np.concatenate(
            [np.linspace(a, b, n, endpoint=False) for a, b, n in zip(start, end, num)]
        )
    else:
        steps = np.array([])

    if endpoint:
        steps = np.append(steps, points[-1])

    if axis is not None:
        if axes is None:
            axes = axis + 1

        steps_1d = steps
        steps = np.zeros((len(steps_1d), axes))
        steps[:, axis] = steps_1d

    return steps

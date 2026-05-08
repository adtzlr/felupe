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

from ..__about__ import __version__ as version
from ._plugin import Plugin


class AnimationWriterPlugin(Plugin):
    r"""A job plugin to write an animation file during job evaluation.

    Parameters
    ----------
    items : list
        The items to plot.
    filename : str, optional
        The filename of the animation file. Default is "animation.gif". Supported
        formats are GIF and MP4 (or any other format supported by PyVista).
    framerate : int, optional
        The framerate of the animation. Default is 5.
    quality : int, optional
        The quality of the movie. Default is 5. This is only used for movie formats,
        not for GIFs.
    zoom_camera : float, optional
        The zoom factor for the camera. Default is 1.0.
    reset_camera : bool, optional
        Whether to reset the camera before each frame. Default is True.
    show_text : bool, optional
        Whether to show text on the plot. Default is True.

    Notes
    -----
    This plugin should be used with ``off_screen=True``. While it is possible to show
    the plotter during the job evaluation, it may flicker.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> boundaries = fem.dof.uniaxial(
        ...     field, clamped=True, return_loadcase=False
        ... )
        >>> umat = fem.NeoHooke(mu=1, bulk=2)
        >>> solid = fem.SolidBody(umat=umat, field=field)
        >>>
        >>> move = fem.math.linsteps([0, 1, 0], num=5)
        >>> ramp = {boundaries["move"]: move}
        >>> step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
        >>>
        >>> animation = fem.AnimationWriterPlugin(
        ...     items=[field],
        ...     filename="animation.gif",
        ...     name="Principal Values of Logarithmic Strain",
        ... )
        >>> job = fem.Job(steps=[step], plugins=[animation])
        >>> job.evaluate()

    See Also
    --------
    felupe.Job : A job with a list of steps and a method to evaluate them.
    """

    def __init__(
        self,
        items,
        filename="animation.gif",
        framerate=5,
        quality=5,
        zoom_camera=1.0,
        reset_camera=True,
        show_text=True,
        **kwargs,
    ):
        self.items = items
        self.filename = filename
        self.framerate = framerate
        self.quality = quality
        self.zoom_camera = zoom_camera
        self.reset_camera = reset_camera
        self.show_text = show_text
        self.kwargs = kwargs

        self.plotter = None

    def _close_plotter(self):
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    def before_job(self, context, state):
        self.plotter = self.kwargs.pop("plotter", None)

        for item in self.items:
            self.plotter = item.plot(plotter=self.plotter, **self.kwargs)

        if self.zoom_camera != 1.0:
            self.plotter.camera.zoom(self.zoom_camera)

        extension = self.filename.split(".")[-1]

        if extension == "gif":
            self.plotter.open_gif(self.filename, fps=self.framerate)

        else:  # "mp4" or any other supported movie format
            self.plotter.open_movie(
                self.filename,
                framerate=self.framerate,
                quality=self.quality,
            )

    def after_iteration(self, context, state):
        if state.error:
            self._close_plotter()

    def after_substep(self, context, state):
        self.plotter.clear_actors()

        for item in self.items:
            self.plotter = item.plot(plotter=self.plotter, **self.kwargs)

        if self.reset_camera:
            self.plotter.reset_camera()

            if self.zoom_camera != 1.0:
                self.plotter.camera.zoom(self.zoom_camera)

        if self.show_text:

            text = [
                f"FElupe {version}",
                f"Step {1 + state.stepnumber}, Substep { 1 + state.substepnumber}",
            ]

            mesh = self.plotter.meshes[0]
            name = mesh.active_scalars_name

            if name is not None:
                min_value = f"{mesh[name].min():.3e}"
                max_value = f"{mesh[name].max():.3e}"

                text.append(f"Min. Value {min_value}")
                text.append(f"Max. Value {max_value}")

            self.plotter.add_text("\n".join(text), font_size=10)

        self.plotter.write_frame()

    def after_job(self, context, state):
        self._close_plotter()

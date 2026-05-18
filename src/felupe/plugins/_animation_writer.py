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
    filename : str or None, optional
        The filename of the animation file. Default is "animation.gif". Supported
        formats are GIF and MP4 (or any other format supported by PyVista). If None, no
        animation file will be written. Default is "animation.gif".
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
    take_screenshots : bool or None, optional
        Whether to take screenshots for all substeps per step. If True, the screenshots
        will be saved in the "figures" folder. Default is False.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the plot method of the items.

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
        take_screenshots=False,
        **kwargs,
    ):
        self.items = items
        self.filename = filename
        self.framerate = framerate
        self.quality = quality
        self.zoom_camera = zoom_camera
        self.reset_camera = reset_camera
        self.show_text = show_text
        self.take_screenshots = take_screenshots
        self.kwargs = kwargs

        self.plotter = None
        self.write_frame = False

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

        if self.filename is not None:

            _, extension = os.path.splitext(self.filename)
            if extension == ".gif":
                self.write_frame = True
                self.plotter.open_gif(self.filename, fps=self.framerate)

            elif extension == ".mp4":
                self.write_frame = True
                self.plotter.open_movie(
                    self.filename,
                    framerate=self.framerate,
                    quality=self.quality,
                )

            else:
                raise TypeError('File extension must be either ".gif" or ".mp4".')

    def after_iteration(self, context, state):
        if state.error:
            self._close_plotter()

    def after_substep(self, context, state):  # pragma: no cover
        self.plotter.clear_actors()  # pragma: no cover

        for item in self.items:  # pragma: no cover
            self.plotter = item.plot(
                plotter=self.plotter, **self.kwargs
            )  # pragma: no cover

        if self.reset_camera:  # pragma: no cover
            self.plotter.reset_camera()  # pragma: no cover

            if self.zoom_camera != 1.0:  # pragma: no cover
                self.plotter.camera.zoom(self.zoom_camera)  # pragma: no cover

        if self.show_text:  # pragma: no cover

            text = [  # pragma: no cover
                f"FElupe {version}",  # pragma: no cover
                f"Step {1 + state.stepnumber}, "  # pragma: no cover
                f"Substep { 1 + state.substepnumber}",  # pragma: no cover
            ]  # pragma: no cover

            for mesh in self.plotter.meshes:  # pragma: no cover
                name = mesh.active_scalars_name  # pragma: no cover
                if name is not None:  # pragma: no cover
                    break  # pragma: no cover

            if name is not None:  # pragma: no cover
                text.append(f"Min. Value {mesh[name].min():.3e}")  # pragma: no cover
                text.append(f"Max. Value {mesh[name].max():.3e}")  # pragma: no cover

            self.plotter.add_text("\n".join(text), font_size=10)  # pragma: no cover

        if self.write_frame:  # pragma: no cover
            self.plotter.write_frame()  # pragma: no cover

        if self.take_screenshots:  # pragma: no cover
            path = "figures/"  # pragma: no cover
            os.makedirs(path, exist_ok=True)  # pragma: no cover

            _ = self.plotter.screenshot(  # pragma: no cover
                path  # pragma: no cover
                + f"step-{(1 + state.stepnumber):02d}_"  # pragma: no cover
                + f"substep-{(1 + state.substepnumber):03d}"  # pragma: no cover
                + ".png"  # pragma: no cover
            )  # pragma: no cover

    def after_job(self, context, state):
        self._close_plotter()

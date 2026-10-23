# SPDX-License-Identifier: GPL-3.0-or-later

from ..view._backends import take_screenshot


class BoundaryDict(dict):
    "A dict of boundary conditions."

    def __iter__(self):
        raise TypeError(
            "Iteration over BoundaryDict is not allowed, use `.items()` instead. "
            "Did you try to run "
            "`boundaries, loadcase = fem.dof.uniaxial(field)`? "
            "Starting with v12.0.0, the default value of `return_loadcase` in "
            "`fem.dof.uniaxial()`, `fem.dof.shear()` and `fem.dof.biaxial()` is False, "
            "i.e. the return of `loadcase` must be explicitly activated by "
            "`return_loadcase=True`."
        )

    def plot(
        self,
        plotter=None,
        colors=None,
        size=(0.1, 0.1),
        show_points=True,
        show_lines=True,
        **kwargs,
    ):
        "Plot the boundary conditions."

        if colors is None:
            import matplotlib.colors as mcolors

            colors = list(mcolors.TABLEAU_COLORS.values())

        colors_list = []
        while len(colors_list) < len(self.keys()):
            colors_list = [*colors_list, *colors]

        for (label, boundary), color in zip(self.items(), colors_list):
            plotter = boundary.plot(
                label=label,
                color=color,
                plotter=plotter,
                size=size,
                show_points=show_points,
                show_lines=show_lines,
                **kwargs,
            )

        plotter.add_legend(size=size, bcolor="white")

        return plotter

    def screenshot(
        self,
        *args,
        filename="boundaries.png",
        transparent_background=None,
        scale=None,
        colors=None,
        plotter=None,
        **kwargs,
    ):
        "Take a screenshot of the boundary conditions."

        def plot(*args, plotter=None, off_screen=False, **kwargs):
            if plotter is None:
                mesh = self[list(self.keys())[0]].field.region.mesh
                plotter = mesh.plot(off_screen=off_screen)

            return self.plot(plotter=plotter, colors=colors, **kwargs)

        return take_screenshot(
            plot,
            *args,
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
            plotter=plotter,
            **kwargs,
        )

    def imshow(self, *args, ax=None, dpi=None, **kwargs):
        """Take a screenshot of the boundary conditions, show the image data in
        a figure and return the ax.
        """

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(dpi=dpi)

        ax.imshow(self.screenshot(*args, filename=None, **kwargs))
        ax.set_axis_off()

        return ax

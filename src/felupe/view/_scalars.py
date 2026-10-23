# SPDX-License-Identifier: GPL-3.0-or-later

"""Labels of the scalars of a scene, i.e. of the named arrays of point- or cell-data.
They are independent of the plotting backend."""

import numpy as np

PRINCIPAL = "Principal Values of "

COMPONENT_LABELS = {
    1: [""],
    2: ["X", "Y"],
    3: ["X", "Y", "Z"],
    6: ["XX", "YY", "ZZ", "XY", "YZ", "XZ"],
    9: ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"],
}

PRINCIPAL_LABELS = {
    2: ["(Max. Principal)", "(Min. Principal)"],
    3: ["(Max. Principal)", "(Int. Principal)", "(Min. Principal)"],
}


def is_principal(name):
    "Return True if the named scalars are principal values."
    return PRINCIPAL in name


def sort_principal_values(values):
    "Return the principal values, sorted in descending order along the last axis."
    return np.flip(np.sort(values, axis=-1), axis=-1)


def component_labels(name, dim):
    """Return the labels of the components of the named scalars with ``dim``
    components, e.g. ``"X"`` for vectors, ``"XY"`` for tensors in full or in reduced
    vector storage (Voigt-notation) and ``"(Max. Principal)"`` for principal values
    (sorted in descending order). Without a known label, the components are numbered.
    """

    labels = COMPONENT_LABELS

    if is_principal(name):
        labels = {**COMPONENT_LABELS, **PRINCIPAL_LABELS}

    return labels.get(dim, np.arange(dim))


def scalars_label(name, component=0, dim=1):
    """Return the label of the scalar bar of a component of the named scalars with
    ``dim`` components, e.g. ``"Displacement X"``. If the component is None, the label
    of the magnitude is returned, e.g. ``"Displacement Magnitude"``. The prefix
    ``"Principal Values of "`` is removed from the names of principal values, e.g.
    ``"Logarithmic Strain (Max. Principal)"``.
    """

    if component is None:
        return f"{name} Magnitude"

    title = name

    if is_principal(name):
        title = name[len(PRINCIPAL) :]

    return f"{title} {component_labels(name, dim)[component]}"


def surface_scalars(scene, surface, name, component=0, label=None):
    """Return the scalars of a scene, selected by name and component, on a
    triangulated surface of the scene.

    Parameters
    ----------
    scene : felupe.view.Scene
        A scene with the attributes ``point_data`` and ``cell_data``.
    surface : felupe.view._surface.Surface
        The triangulated surface of the mesh of the scene.
    name : str
        The name of the point- or cell-data.
    component : int or None, optional
        The component of the data (default is 0). If None, the magnitude is taken.
    label : str or None, optional
        The label of the scalars (default is None). If None, it is created by
        :func:`scalars_label`.

    Returns
    -------
    values : ndarray
        The values at the points of the surface (point-data) or at the triangles of
        the surface (cell-data).
    where : str
        Where the values are given, ``"point"`` or ``"cell"``.
    values_range : list of float or None
        The range ``[min, max]`` of the finite values of all points or cells of the
        scene (not only of the surface), or None if there are no finite values.
    label : str
        The label of the scalars.

    Raises
    ------
    KeyError
        If no point- or cell-data with the given name is available.
    """

    if name in scene.point_data:
        where, data = "point", scene.point_data[name]
    elif name in scene.cell_data:
        where, data = "cell", scene.cell_data[name]
    else:
        available = ", ".join([*scene.point_data, *scene.cell_data])
        raise KeyError(f'"{name}" is not available, choose one of {available}.')

    data = np.asarray(data, dtype=float)
    dim = data.shape[1] if data.ndim == 2 else 1
    data = data.reshape(len(data), -1)

    if component is None:
        values = np.linalg.norm(data, axis=1)
    else:
        if is_principal(name):
            data = sort_principal_values(data)
        values = data[:, component]

    finite = values[np.isfinite(values)]
    values_range = [float(finite.min()), float(finite.max())] if finite.size else None

    if where == "point":
        values = surface.interpolate(values)
    else:
        values = values[surface.owners]

    if label is None:
        label = scalars_label(name, component=component, dim=dim)

    return values, where, values_range, label

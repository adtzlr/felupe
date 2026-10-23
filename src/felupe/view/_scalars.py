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

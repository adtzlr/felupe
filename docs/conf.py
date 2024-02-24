# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "FElupe"
copyright = "2024, Andreas Dutzler"
author = "Andreas Dutzler"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

nb_execution_mode = "off"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Execution mode for notebooks
# nb_execution_mode = "force"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_title = "FElupe"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "icon_links": [
        {
            "name": "Discussions",
            "url": "https://github.com/adtzlr/felupe/discussions",
            "icon": "fa-brands fa-comment",
            "type": "fontawesome",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/adtzlr/felupe",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Read the Docs",
            "url": "https://readthedocs.org/projects/felupe/downloads",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/felupe/",
            "icon": "fa-solid fa-box",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": "FElupe",
        "image_light": "logo_light.svg",
        "image_dark": "logo_dark.svg",
    },
}

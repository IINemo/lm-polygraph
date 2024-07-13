# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "LM-Polygraph"
author = "MBZUAI"
copyright = f"{datetime.now().year}, {author}"

# release = setuptools_scm.get_version(root='..', relative_to=__file__)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = ["Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"

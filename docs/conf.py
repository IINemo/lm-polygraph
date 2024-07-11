# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import subprocess
import sys
from pathlib import Path
import setuptools_scm
from datetime import datetime


# Get the absolute path of the root directory
root_path = Path(__file__).resolve().parents[2]

# Add the src/lm_polygraph directory to the Python path
sys.path.insert(0, str(root_path / "src" / "lm_polygraph"))
sys.path.insert(0, str(root_path))

# Install the library using pip
subprocess.run(["pip", "install", "-e", "."], cwd=root_path)


project = "LM-Polygraph"
author = "MBZUAI"
copyright = f'{datetime.now().year}, {author}'

release = setuptools_scm.get_version(root='..', relative_to=__file__)

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
html_theme = 'sphinx_rtd_theme'
html_static_path = ["_static"]

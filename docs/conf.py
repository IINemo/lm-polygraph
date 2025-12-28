# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

# -- Auto-generate API documentation -----------------------------------------
# Run sphinx-apidoc automatically when building docs

def run_apidoc(_):
    from sphinx.ext.apidoc import main as apidoc_main
    
    docs_dir = Path(__file__).parent
    output_dir = docs_dir / "api"  # Output .rst files to docs/
    source_dir = docs_dir.parent / "src" / "lm_polygraph"
    
    # Only run if source directory exists
    if source_dir.exists():
        apidoc_main([
            "--force",  # Overwrite existing files
            "--module-first",  # Put module documentation before submodule documentation
            "--separate",  # Create separate pages for each module
            "-o", str(output_dir),
            str(source_dir),
        ])

def setup(app):
    app.connect("builder-inited", run_apidoc)

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
latex_engine = "xelatex"

# -- Autosummary settings ----------------------------------------------------
autosummary_generate = True  # Auto-generate stub pages for all modules

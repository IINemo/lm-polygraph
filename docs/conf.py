# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import subprocess
import sys
from pathlib import Path

# Assuming conf.py is located in /Users/sergeypetrakov/Documents/Documents/Skoltech_PhD/rtd_to_main/lm-polygraph/src/lm_polygraph/docs

# Get the absolute path of the root directory
root_path = Path(__file__).resolve().parents[2]

# Add the src/lm_polygraph directory to the Python path
sys.path.insert(0, str(root_path / "src" / "lm_polygraph"))
sys.path.insert(0, str(root_path))

# Install the library using pip
subprocess.run(["pip", "install", "-e", "."], cwd=root_path)


project = "LM-Polygraph"
copyright = "2023, MBZUAI"
author = "MBZUAI"
release = "0.3.0"

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
exclude_patterns = ["Thumbs.db", ".DS_Store"]  # '_build',
# autodoc_mock_imports = ['lm_polygraph']

# , 'datasets>=2.14.2', "rouge-score>=0.0.4", "nlpaug>=1.1.10",
#                         "scikit-learn", "tqdm>=4.64.1", "matplotlib>=3.6", "pandas>=1.3.5", "torch>=1.13.0",
#                         "bs4", "transformers>=4.30.2", "nltk>=3.6.5", "sacrebleu>=1.5.0", "sentencepiece>=0.1.97",
#                         "hf-lfs==0.0.3", "pytest>=4.4.1", "pytreebank>=0.2.7", "setuptools>=60.2.0",
#                         "numpy>=1.23.5", "dill>=0.3.5.1", "scipy>=1.9.3", "flask>=2.3.2", "protobuf>=4.23",
#                         "einops", "accelerate", "bitsandbytes", "openai", "wget"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_static_path = ["_static"]

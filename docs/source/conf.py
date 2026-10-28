# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Add package root so imports like ``import luas`` resolve correctly.
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'luas'
copyright = '2023, Mark Fortune'
author = 'Mark Fortune'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Packages listed here are replaced by empty placeholder ("mock") modules while
# Sphinx imports luas to read its docstrings. Only list optional packages or
# ones that cannot be installed from PyPI. Do NOT add numpy, jax or scipy:
# they are real dependencies, and scipy crashes on import if numpy is a
# placeholder, which stops ``import luas`` working and leaves the API pages empty.
#   - tinygp: luas uses ``tinygp.solvers.quasisep.block``, which is not in the
#     released tinygp versions on PyPI.
#   - numpyro, pymc, pytensor, aesara: optional inference backends.
#   - jaxoplanet, ldtk, corner: optional extras used by a few helper functions.
autodoc_mock_imports = [
    'tinygp',
    'numpyro',
    'pymc',
    'pytensor',
    'aesara',
    'jaxoplanet',
    'ldtk',
    'corner',
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]
myst_enable_extensions = ["dollarmath", "colon_fence"]
# Generate a stub page for every entry in an ``.. autosummary::`` table with a
# ``:toctree:`` option. Stubs are written to docs/source/summary/ (git-ignored)
# and regenerated on every build.
autosummary_generate = True
# List class members in the order they appear in the source code.
autodoc_member_order = 'bysource'
templates_path = ['_templates']
exclude_patterns = [
    '**/.ipynb_checkpoints',
    # Hidden from the docs for now. Delete these lines and add the pages back
    # to a toctree in api.md to show them again.
    'LuasLasrachPlusPeriodicKernel.rst',
    'BlockKernel.rst',
    # Tutorials hidden for now. Delete these lines (and add the NumPyro example
    # back to tutorials.md) to show them again.
    'tutorials/Numpyro Example.ipynb',
    'tutorials/PyMC Example 1D GP.ipynb',
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'luas'
#html_logo = "_static/luas_icon.png"
html_static_path = ['_static']
html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/markfortune/luas",
    "repository_branch": "main",
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "show_navbar_depth": 2,
    "home_page_in_toc": True,
    "use_edit_page_button": False,
    "use_issues_button": False,
    "use_repository_button": True,
    "use_download_button": True,
}
nb_execution_mode = "auto"
nb_execution_timeout = 60
nb_merge_streams = True
nb_execution_excludepatterns = ["2D_GP_intro.ipynb", "Numpyro Example.ipynb", "PyMC Example.ipynb"]

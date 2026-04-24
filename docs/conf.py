"""Sphinx configuration for combss documentation."""

project = "combss"
copyright = "2024, Sarat Moka, Anant Mathur, Hua Yang Hu"
author = "Sarat Moka, Anant Mathur, Hua Yang Hu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# General settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "../combss_logo.png"
html_theme_options = {
    "logo_only": False,
}

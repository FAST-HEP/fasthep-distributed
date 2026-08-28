from __future__ import annotations

project = "FAST-HEP Distributed"
author = "FAST-HEP contributors"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/FAST-HEP/fasthep-distributed",
    "logo": {
        "text": "FAST-HEP Distributed",
    },
    "navbar_align": "left",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]

exclude_patterns = [
    "_build",
]

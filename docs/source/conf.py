# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinxcontrib.katex as katex

project = "dmme"
copyright = "2022, Chanhyuk Jung"
author = "Chanhyuk Jung"
release = "0.5.2-beta0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]

myst_enable_extensions = ["dollarmath"]

autodoc_member_order = "bysource"

latex_macros = r"""
    \def \bb                    #1{{\mathbb{#1}}}
    \def \bepsilon              {\boldsymbol{\epsilon}}
    \def \bmu                   {\boldsymbol{\mu}}
    \def \bx                    {\boldsymbol{x}}
    \def \bzero                 {\boldsymbol{0}}
    \def \bI                    {\boldsymbol{I}}
    \def \bSigma                {\boldsymbol{\Sigma}}
    \def \defeq                 {\coloneqq}
    \def \E                     {\mathbb{E}}
    \def \Ea                  #1{\E\left[#1\right]}
    \def \Eb                #1#2{\E_{#1}{\!\left[#2\right]}}
    \def \gQ                    {\mathcal{Q}}
    \def \gN                    {\mathcal{N}}
    \def \kl                #1#2{D_{\mathrm{KL}}\!\left(#1 ~ \| ~ #2\right)}
    \def \mI                    {{\bm{I}}}
    \def \R                     {\mathbb{R}}
    \def \vone                  {{\bm{1}}}
    \def \vx                    {\boldsymbol{x}}
    \def \vzero                 {\bm{0}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = "macros: {" + katex_macros + "}"

# Add LaTeX macros for LATEX builder
latex_elements = {"preamble": latex_macros}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

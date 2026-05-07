# Configuration file for the Sphinx documentation builder.
# This file uses modern Sphinx 9.0+ configuration with PyData Sphinx Theme
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from datetime import datetime
from importlib.metadata import metadata

# Import the example gallery generator
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import _example_gallery
generate_examples_gallery = _example_gallery.generate_examples_gallery

# Add custom extensions directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '_ext')))

# Add the project root to sys.path to enable autodoc to find the package
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Read metadata from installed package
pkg_metadata = metadata('causallib')
project = pkg_metadata['Name']
author = pkg_metadata['Author']
copyright = f'2017-{datetime.now().year}, CausalML for HCLS; IBM Research ISRL'
release = pkg_metadata["version"]
version = pkg_metadata["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',           # Automatic documentation from docstrings
    'sphinx.ext.autosummary',       # Generate autodoc summaries
    'sphinx.ext.doctest',           # Test snippets in documentation
    'sphinx.ext.coverage',          # Documentation coverage
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx.ext.intersphinx',       # Link to other project's documentation
    'sphinx.ext.napoleon',          # Support for NumPy and Google style docstrings
    'sphinx.ext.mathjax',           # Render math via JavaScript
    
    # Third-party extensions
    'numpydoc',                     # NumPy-style docstring support
    'sphinx_design',                # Design elements (cards, tabs, etc.)
    'sphinx_copybutton',            # Add copy button to code blocks
    'myst_nb',                      # Jupyter notebook and Markdown support via MyST
    
    # Custom extensions
    'type_crossref',                # Convert type aliases to cross-references
]

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    # 'no-index': True,  # Avoid duplicate classes warnings but breaks hyper-linking
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autoclass_content = 'both'  # Include both class and __init__ docstrings

# Type aliases for autodoc to resolve abbreviated type names
autodoc_type_aliases = {
    'pd.DataFrame': 'pandas.DataFrame',
    'pd.Series': 'pandas.Series',
    'pd.Index': 'pandas.Index',
    'np.ndarray': 'numpy.ndarray',
    'np.array': 'numpy.ndarray',

    'Any': 'typing.Any',
    'IndividualOutcomeEstimator': 'causallib.estimation.base_estimator.IndividualOutcomeEstimator',
    'WeightEstimator': 'causallib.estimation.base_weight.WeightEstimator',
}

# Mock imports for optional dependencies that may not be installed
autodoc_mock_imports = [
    'faiss',
    'torch',
    'aix360',
    'rpy2',
    'mlxtend',
    'formulaic',
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False
autosummary_recursive = True  # Recursively discover all modules
add_module_names = False

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'pd': 'pandas',
    'DataFrame': 'pandas.DataFrame',
    'Series': 'pandas.Series',
    'np': 'numpy',
    'ndarray': 'numpy.ndarray',
    'array': 'numpy.ndarray',
}
napoleon_attr_annotations = True

# NumPy doc settings
numpydoc_show_class_members = False  # Suppress warnings when building numpy-doc
numpydoc_class_members_toctree = False

# Intersphinx mapping - links to other projects' documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
}

# myst-nb settings for Jupyter notebooks
nb_execution_mode = 'off'  # Don't execute notebooks during build
nb_execution_allow_errors = True
nb_kernel_rgx_aliases = {"python3": "python3"}

# MyST parser settings for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

# PyData theme options
html_theme_options = {
    "logo": {
        "text": "causallib",
        "alt_text": "causallib - Home",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BiomedSciAI/causallib",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/causallib/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "primary_sidebar_end": ["sidebar-ethical-ads"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "show_prev_next": True,
    "show_nav_level": 2,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "use_edit_page_button": True,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "pygment_light_style": "default",
    "pygment_dark_style": "monokai",
}

# Edit on GitHub configuration
html_context = {
    "github_user": "BiomedSciAI",
    "github_repo": "causallib",
    "github_version": "master",
    "doc_path": "docs/source",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS and JavaScript files
html_css_files = [
    'custom.css',
]

html_js_files = [
    # 'js/custom.js',  # Uncomment when custom JS is added
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/logo.png'  # Uncomment when logo is added

# The name of an image file (within the static path) to use as favicon of the
# docs. This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = '_static/favicon.ico'  # Uncomment when favicon is added

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Options for source files ------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# Note: myst-nb automatically registers .md and .ipynb suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
}

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    '*estimation.*_.rst',  # Old estimation files kept locally
    '*tests.*',            # Remove tests from documentation
]

# Suppress warnings that are acceptable for this documentation
suppress_warnings = [
    'toc.not_included',      # Auto-generated API pages don't need to be in toctree
    'ref.citation',          # Unreferenced citations in docstrings
    'autosummary',           # Suppress autosummary warnings
    'autodoc.import_object', # Suppress import warnings
]

# ### There are 1500+ warnings of type "duplicate object description" that confuse sphinx imports
# ### They are caused due to classes being imported multiple times due to __init__ shortcuts, like:
# ### `estimation.ipw.IPW` and `estimation.IPW`.
# ### This is a workaround to suppress them
# # Custom warning filter to suppress duplicate object warnings
# import logging

# # Create a custom filter at module level
# class DuplicateObjectFilter(logging.Filter):
#     """Filter to suppress 'duplicate object description' warnings."""
#     def filter(self, record):
#         message = record.getMessage()
#         # Suppress duplicate object description warnings
#         if 'duplicate object description' in message:
#             return False
#         # Suppress "don't know which module to import" warnings
#         if "don't know which module to import" in message:
#             return False
#         return True

# # Apply the filter to sphinx loggers immediately
# for logger_name in ['sphinx', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']:
#     logger = logging.getLogger(logger_name)
#     logger.addFilter(DuplicateObjectFilter())

# def setup(app):
#     """Sphinx setup hook to ensure filters are applied."""
#     # Re-apply filters in case loggers were recreated
#     duplicate_filter = DuplicateObjectFilter()
#     for logger_name in ['sphinx', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']:
#         logger = logging.getLogger(logger_name)
#         logger.addFilter(duplicate_filter)

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',
    
    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '10pt',
    
    # Additional stuff for the LaTeX preamble.
    'preamble': '',
    
    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'causallib.tex', 'causallib Documentation',
     author, 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'causallib', 'causallib Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'causallib', 'causallib Documentation',
     author, 'causallib', 'A Python package for flexible and modular causal inference modeling',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# -- Custom scripts ----------------------------------------------------------

def add_modules_readme():
    """
    Add module's README for each module HTML page.
    This function preserves the original behavior from the old conf.py.
    """
    def get_rst_file_name_from_package_source(package_source_path):
        rst_file_name = package_source_path.split(os.sep)
        rst_file_name = [directory for directory in rst_file_name if not directory.startswith('.')]
        rst_file_name += ['rst']
        rst_file_name = ".".join(rst_file_name)
        rst_file_name = os.path.join(source_html_dir, rst_file_name)
        return rst_file_name

    def get_edited_rst_file(rst_source_path, include_text, remove_original_header=True):
        with open(rst_source_path, 'r') as fh:
            rst_source_lines = fh.read().splitlines()

        if not any([include_text in line for line in rst_source_lines]):
            # Add link to readme only if not already exists
            rst_source_lines.insert(3, include_text)
            if remove_original_header:
                rst_source_lines = rst_source_lines[2:]  # remove existing header

        return rst_source_lines

    INCLUDE_TEXT = ".. mdinclude:: "
    README_FILE_NAME = 'README.md'
    
    source_code_dir = os.path.join("..", "..", "causallib")  # causallib source code directory
    source_html_dir = "."  # sphinx's docs source directory
    
    for dir_name, subdir_list, file_names in os.walk(source_code_dir):
        if README_FILE_NAME in file_names:  # Current dir has a readme file
            # Get README file path:
            source_path = os.path.normpath(dir_name)
            readme_file_path = os.path.join(source_path, README_FILE_NAME)

            # Construct the corresponding module rst file:
            rst_source_file = get_rst_file_name_from_package_source(source_path)

            # Edit the rst file to include the path to the readme:
            try:
                include_text = INCLUDE_TEXT + readme_file_path + "\n"
                content = get_edited_rst_file(rst_source_file, include_text, True)

                with open(rst_source_file, 'w') as f:
                    for line in content:
                        f.write("{}\n".format(line))

            except FileNotFoundError:
                print("Could not find file {}".format(rst_source_file))


# Execute custom scripts
add_modules_readme()
generate_examples_gallery()


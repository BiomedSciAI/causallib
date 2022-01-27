# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


# -- Project information -----------------------------------------------------

# Take from setup.py file
def grep_value_from_setup(key):
	import re
	setup_file = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "setup.py"))
	with open(setup_file, 'r') as setup_file:
		for line in setup_file.readlines():
			value = re.search(".*{key}=(.*),*".format(key=key), line)
			if value:
				value = value.group(1)
				value = re.sub(r'^\W+', '', value)  # lstrip non alphanumeric
				value = re.sub(r'\W+$', '', value)  # rstrip non alphanumeric
				return value


project = grep_value_from_setup('name')
copyright = '2019, IBM Research Haifa Labs - MLHLS'
author = grep_value_from_setup('author')

# The full version, including alpha/beta/rc tags
from causallib import __version__
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',	
	'sphinx.ext.doctest',
	'sphinx.ext.coverage',
	'sphinx.ext.viewcode',
	'sphinx.ext.intersphinx',  # link to other documentations
	# 'nbsphinx',  # work with jupyter notebook. requires pip install nbsphinx
	# 'sphinx.ext.mathjax',		# required for math in Jupyter Notebook
	'm2r2',  # work with markdown files. requires pip install m2r2
	'sphinx.ext.napoleon',  # parse Google/numpy docstring
	# 'sphinx.ext.autosummary',	# Automated index.rst toctree
]

# autosummary_generate = True		# create api docs with autosummary


intersphinx_mapping = {
	'numpy': ('http://docs.scipy.org/doc/numpy/', None),
	'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
	'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
	'python': ('https://docs.python.org/{}.{}'.format(*sys.version_info), None)
}

autoclass_content = 'both'

numpydoc_show_class_members = False	 # Suppress sphinx warnings when building numpy-doc

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
	'.rst': 'restructuredtext',
	'.txt': 'restructuredtext',
	'.md': 'markdown',
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
	'*estimation.*_.rst',		# estimation old files which kept locally
	'*tests.*'					# remove tests from documentation
]

# https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=master_doc#confval-master_doc
master_doc = 'index'    # what file includes the root toctree. Needed explicitly for readthedocs.org


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
	html_theme = 'sphinx_rtd_theme'
else:
	html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# # Add content to the navigation side bar:
html_sidebars = {'**': ['localtoc.html', 'sourcelink.html', 'searchbox.html', 'globaltoc.html']}


# -- Custom scripts -------------------------------------------------

# Add module's README for each module HTML page
def add_modules_readme():
	def get_rst_file_name_from_package_source(package_source_path):
		rst_file_name = package_source_path.split(os.sep)
		rst_file_name = [directory for directory in rst_file_name if not directory.startswith('.')]
		rst_file_name += ['rst']
		rst_file_name = ".".join(rst_file_name)
		rst_file_name = os.path.join(source_html_dir, rst_file_name)
		return rst_file_name

	def get_edited_rast_file(rst_source_path, include_text, remove_original_header=True):
		with open(rst_source_path, 'r') as fh:
			rst_source_lines = fh.read().splitlines()

		if not any([include_text in line for line in rst_source_lines]):
			# Add link to readme only if not already exists
			rst_source_lines.insert(3, include_text)
			if remove_original_header:
				rst_source_lines = rst_source_lines[2:]  # remove existing header

		return rst_source_lines

	INCLUDE_TEXT = ".. mdinclude:: "
	# INCLUDE_TEXT += "../../"
	
	README_FILE_NAME = 'README.md'
	
	source_code_dir = os.path.join("..", "..", "causallib")  # causallib source code directory
	source_html_dir = "."  # sphinx's docs source directory
	
	for dir_name, subdir_list, file_names in os.walk(source_code_dir):
		if README_FILE_NAME in file_names:	# Current dir has a readme file
			# Get README file path:
			# source_path = os.path.relpath(dir_name, os.path.join(source_code_dir, ".."))
			source_path = os.path.normpath(dir_name)
			readme_file_path = os.path.join(source_path, README_FILE_NAME)

			# Construct the corresponding module rst file:
			rst_source_file = get_rst_file_name_from_package_source(source_path)

			# Edit the rst file to include the path to the readme:
			try:
				include_text = INCLUDE_TEXT + readme_file_path + "\n"
				content = get_edited_rast_file(rst_source_file, include_text, True)

				with open(rst_source_file, 'w') as f:
					for line in content:
						f.write("{}\n".format(line))

			except FileNotFoundError:
				print("Could not find file {}".format(rst_source_file))


add_modules_readme()


"""
Custom Sphinx extension to convert type aliases in docstrings to proper cross-references.
This handles cases where Napoleon processes 'pd.DataFrame' as plain text.
"""

import re
from sphinx.application import Sphinx


def process_docstring(app, what, name, obj, options, lines):
    """
    Process docstring lines to replace type aliases with proper cross-references.
    This runs before Napoleon processes the docstring.
    """
    # Define type mappings - map abbreviated forms to full module paths
    type_mappings = {
        'pd.DataFrame': 'pandas.DataFrame',
        'pd.Series': 'pandas.Series',
        'pd.Index': 'pandas.Index',
        'np.ndarray': 'numpy.ndarray',
        'np.array': 'numpy.ndarray',
    }
    
    for i, line in enumerate(lines):
        for alias, replacement in type_mappings.items():
            # Replace standalone type aliases (not part of longer words)
            # This needs to handle both parameter types and return types
            pattern = r'\b' + re.escape(alias) + r'\b'
            lines[i] = re.sub(pattern, replacement, lines[i])


def setup(app: Sphinx):
    """Setup the extension."""
    # Connect early in the process, before Napoleon
    app.connect('autodoc-process-docstring', process_docstring, priority=0)
    
    return {
        'version': '0.2',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

# Made with Bob

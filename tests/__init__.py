"""
Tests package for OpenCSP.

This file sets up the context for pytest to allow running tests
without installing the package.
"""

import os
import sys
from pathlib import Path

# Get the project root directory (parent of tests directory)
project_root = Path(__file__).parent.parent.absolute()
print(project_root)

# Add the project root to sys.path if it's not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import opencsp to verify the context is set correctly
try:
    import opencsp
    print(f"OpenCSP package found at: {opencsp.__file__}")
except ImportError:
    raise ImportError(
        "OpenCSP package not found. Make sure your project structure is correct."
    )

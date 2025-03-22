"""
Pytest configuration file for OpenCSP tests.

This file contains fixtures and configuration for pytest that can be
shared across all test modules.
"""

import os
import sys
import pytest
from pathlib import Path

# Get the project root directory (parent of tests directory)
project_root = Path(__file__).parent.parent.absolute()

# Add the project root to sys.path if it's not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))



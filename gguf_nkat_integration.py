"""Thin wrapper to import integration tools from the ``scripts`` directory."""

import os
import sys

# Allow importing ``scripts`` when running tests from the ``tests`` directory.
_base_dir = os.path.dirname(__file__)
project_root = _base_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.gguf_nkat_integration import *  # noqa: F401,F403


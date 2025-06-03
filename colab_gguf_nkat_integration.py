"""Wrapper to import Colab integration helpers from ``scripts`` directory."""

import os
import sys

_base_dir = os.path.dirname(__file__)
project_root = _base_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.colab_gguf_nkat_integration import *  # noqa: F401,F403


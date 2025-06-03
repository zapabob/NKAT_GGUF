"""Test wrapper for importing Colab helper from project scripts."""

import os
import sys

_base_dir = os.path.dirname(os.path.dirname(__file__))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

from scripts.colab_gguf_nkat_integration import *  # noqa: F401,F403


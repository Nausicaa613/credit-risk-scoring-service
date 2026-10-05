"""Support package for the command-line entry points.

``riskscore`` lives under ``src/`` and is not installed by default, so adding it
to ``sys.path`` here keeps the scripts runnable straight from a checkout.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

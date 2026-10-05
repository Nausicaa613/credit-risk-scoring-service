"""Make the repository importable in tests without an install step.

Running ``python -m unittest discover -s tests`` from a fresh checkout must work
with no virtualenv, no ``pip install -e .`` and no ``PYTHONPATH`` fiddling. A
``sitecustomize``-style hack would be fragile, so the path is adjusted here in
the one module unittest always imports first.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Keep tests hermetic: nothing should read or write the developer's real config.
os.environ.setdefault("RISKSCORE_LOG_LEVEL", "CRITICAL")

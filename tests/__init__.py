"""Test package.

The package initialiser puts ``src/`` on ``sys.path`` so the suite runs from a
fresh checkout with no install step and no ``PYTHONPATH`` configuration:

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Several tests deliberately exercise degraded paths -- a missing model artifact,
# a broken audit sink -- and the service correctly logs a warning or a traceback
# for each. That is the behaviour under test, not a problem, so keep it out of
# the test output. Individual tests re-enable logging via assertLogs when the log
# record itself is the thing being asserted.
logging.disable(logging.CRITICAL)
logging.getLogger().addHandler(logging.NullHandler())
logging.getLogger().lastResort = None

os.environ.setdefault("RISKSCORE_LOG_LEVEL", "CRITICAL")

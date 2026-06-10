"""Make ``src.motion`` importable when running pytest from the repo root.

OpenSceneFlow has no installable package; scripts bootstrap sys.path themselves
(see tools/detect_track_movers.py). Mirror that here so tests can import src.*.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

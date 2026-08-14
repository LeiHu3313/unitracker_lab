"""Load simulator-independent G1 teacher modules without importing the extension package."""

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = (
    Path(__file__).resolve().parents[2]
    / "source"
    / "unitracker_lab"
    / "unitracker_lab"
    / "tasks"
    / "manager_based"
    / "unitracker_teacher"
)
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASK_DIR / "mdp"))

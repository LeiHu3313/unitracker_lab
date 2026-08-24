from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "source/unitracker_lab/unitracker_lab/tasks/manager_based/extreme_rgmt"
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASK_DIR / "mdp"))

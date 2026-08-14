"""Validate prepared G1 Stage-1 motion data without launching Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_DIR = (
    _REPO_ROOT / "source" / "unitracker_lab" / "unitracker_lab" / "tasks" / "manager_based" / "unitracker_teacher"
)
sys.path.insert(0, str(_SCHEMA_DIR))
from motion_schema import load_and_validate_motion_dataset  # noqa: E402


def main(path: Path) -> None:
    motion = load_and_validate_motion_dataset(path)
    print("VALID G1 STAGE-1 MOTION DATASET")
    print(json.dumps(motion.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    arguments = parser.parse_args()
    main(arguments.motion)

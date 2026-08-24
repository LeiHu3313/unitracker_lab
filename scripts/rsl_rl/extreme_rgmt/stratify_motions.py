#!/usr/bin/env python3
"""Freeze Extreme-RGMT mastered/challenging manifests from five-rollout results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_record(record: dict, source: Path) -> dict:
    path = Path(record["path"]).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    path = path.resolve()
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise ValueError(f"Invalid motion path: {path}")
    successes = record.get("successes")
    if not isinstance(successes, list) or len(successes) != 5 or not all(isinstance(x, bool) for x in successes):
        raise ValueError(f"{path}: successes must be a list of exactly five booleans.")
    completion_rate = sum(successes) / len(successes)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "successes": successes,
        "completion_rate": completion_rate,
        "role": "mastered" if completion_rate >= 0.8 else "challenging",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSON with a top-level 'clips' list.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.results.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = [_parse_record(record, source) for record in payload.get("clips", [])]
    if not records:
        raise ValueError("No clip results were provided.")
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Stratification input contains duplicate motion paths.")

    mastered = [record for record in records if record["role"] == "mastered"]
    challenging = [record for record in records if record["role"] == "challenging"]
    if not mastered or not challenging:
        raise ValueError("Both mastered and challenging sets must be non-empty.")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "mastered.lst").write_text("".join(f"{record['path']}\n" for record in mastered), encoding="utf-8")
    (output / "challenging.lst").write_text("".join(f"{record['path']}\n" for record in challenging), encoding="utf-8")
    report = {
        "paper": "arXiv:2607.20110v1",
        "source_results": str(source),
        "rollouts_per_clip": 5,
        "mastered_threshold": 0.8,
        "mastered_count": len(mastered),
        "challenging_count": len(challenging),
        "clips": records,
    }
    (output / "stratification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

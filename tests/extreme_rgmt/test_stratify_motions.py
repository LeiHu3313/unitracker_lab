from __future__ import annotations

from scripts.rsl_rl.extreme_rgmt.stratify_motions import _parse_record


def test_stratification_uses_five_rollouts_and_paper_threshold(tmp_path):
    source = tmp_path / "results.json"
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"fixture")

    mastered = _parse_record({"path": "motion.npz", "successes": [True, True, True, True, False]}, source)
    challenging = _parse_record({"path": "motion.npz", "successes": [True, True, True, False, False]}, source)

    assert mastered["completion_rate"] == 0.8
    assert mastered["role"] == "mastered"
    assert challenging["completion_rate"] == 0.6
    assert challenging["role"] == "challenging"

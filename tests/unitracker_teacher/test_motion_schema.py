from pathlib import Path

import numpy as np
import pytest
from contracts import G1_ALL_JOINT_NAMES, G1_CONTROLLED_JOINT_NAMES, G1_LOCKED_WRIST_JOINT_NAMES, G1_TRACKING_BODY_NAMES
from motion_schema import load_and_validate_motion, load_and_validate_motion_dataset, resolve_motion_files


def _write_motion(
    path: Path,
    *,
    fps: float = 50.0,
    frames: int = 3,
    value: float = 0.0,
    joint_names=None,
    body_names=None,
):
    joint_names = tuple(joint_names or G1_CONTROLLED_JOINT_NAMES)
    body_names = tuple(body_names or G1_TRACKING_BODY_NAMES)
    joint = np.full((frames, len(joint_names)), value, dtype=np.float32)
    body = np.full((frames, len(body_names), 3), value, dtype=np.float32)
    quat = np.zeros((frames, len(body_names), 4), dtype=np.float32)
    quat[..., 0] = 1.0
    np.savez(
        path,
        fps=np.asarray(fps),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        joint_pos=joint,
        joint_vel=joint,
        body_pos_w=body,
        body_quat_w=quat,
        body_lin_vel_w=body,
        body_ang_vel_w=body,
    )


def test_loader_reorders_names_into_checkpoint_contract(tmp_path):
    path = tmp_path / "motion.npz"
    _write_motion(
        path,
        joint_names=tuple(reversed(G1_CONTROLLED_JOINT_NAMES)),
        body_names=tuple(reversed(G1_TRACKING_BODY_NAMES)),
    )
    motion = load_and_validate_motion(path)
    assert motion.joint_pos.shape == (3, 23)
    assert motion.body_pos_w.shape == (3, 16, 3)
    assert motion.frame_count == 3


def test_loader_rejects_non_control_rate_motion(tmp_path):
    path = tmp_path / "motion_60hz.npz"
    _write_motion(path, fps=60.0)
    with pytest.raises(ValueError, match="50.0 Hz"):
        load_and_validate_motion(path)


def test_loader_requires_explicit_names(tmp_path):
    path = tmp_path / "missing_names.npz"
    _write_motion(path)
    with np.load(path, allow_pickle=False) as data:
        values = {name: data[name] for name in data.files if name != "body_names"}
    np.savez(path, **values)
    with pytest.raises(ValueError, match="missing required fields"):
        load_and_validate_motion(path)


def test_29_joint_motion_is_reduced_and_locked_wrists_are_validated(tmp_path):
    path = tmp_path / "motion_29.npz"
    _write_motion(path, joint_names=G1_ALL_JOINT_NAMES)
    motion = load_and_validate_motion(path)
    assert motion.joint_pos.shape == (3, 23)

    with np.load(path, allow_pickle=False) as data:
        values = {name: data[name] for name in data.files}
    wrist_id = list(G1_ALL_JOINT_NAMES).index(G1_LOCKED_WRIST_JOINT_NAMES[0])
    values["joint_pos"] = values["joint_pos"].copy()
    values["joint_pos"][0, wrist_id] = 0.2
    np.savez(path, **values)
    with pytest.raises(ValueError, match="wrist-lock"):
        load_and_validate_motion(path)


def test_dataset_recursively_sorts_and_preserves_clip_boundaries(tmp_path):
    _write_motion(tmp_path / "b.npz", frames=4, value=2.0)
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_motion(nested / "a.npz", frames=3, value=1.0)

    dataset = load_and_validate_motion_dataset(tmp_path)

    assert tuple(path.name for path in dataset.paths) == ("b.npz", "a.npz")
    np.testing.assert_array_equal(dataset.clip_starts, [0, 4])
    np.testing.assert_array_equal(dataset.clip_lengths, [4, 3])
    np.testing.assert_array_equal(dataset.clip_ends, [4, 7])
    assert dataset.num_motions == 2
    assert dataset.frame_count == 7
    np.testing.assert_allclose(dataset.joint_pos[:4], 2.0)
    np.testing.assert_allclose(dataset.joint_pos[4:], 1.0)


def test_manifest_supports_relative_paths_comments_and_explicit_order(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    _write_motion(motion_dir / "first.npz")
    _write_motion(motion_dir / "second.npz")
    manifest = tmp_path / "train.lst"
    manifest.write_text(
        "# selected training motions\nmotions/second.npz  # keep manifest order\n\nmotions/first.npz\n",
        encoding="utf-8",
    )

    source, paths = resolve_motion_files(manifest)
    assert source == manifest.resolve()
    assert tuple(path.name for path in paths) == ("second.npz", "first.npz")


def test_dataset_rejects_empty_directory_and_duplicate_manifest_entries(tmp_path):
    with pytest.raises(ValueError, match="No .npz motion files"):
        load_and_validate_motion_dataset(tmp_path)

    path = tmp_path / "motion.npz"
    _write_motion(path)
    manifest = tmp_path / "duplicate.txt"
    manifest.write_text("motion.npz\nmotion.npz\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate file paths"):
        load_and_validate_motion_dataset(manifest)


def test_dataset_reports_the_invalid_clip_path(tmp_path):
    good = tmp_path / "good.npz"
    bad = tmp_path / "bad.npz"
    _write_motion(good)
    _write_motion(bad, fps=60.0)

    with pytest.raises(ValueError, match=r"Invalid G1 motion clip .*bad\.npz"):
        load_and_validate_motion_dataset(tmp_path)

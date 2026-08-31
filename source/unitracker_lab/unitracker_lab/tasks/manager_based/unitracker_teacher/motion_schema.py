"""Strict, simulator-independent loaders for prepared G1 reference motions."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .contracts import (
        CONTROL_FREQUENCY_HZ,
        G1_ALL_JOINT_NAMES,
        G1_MOTION_BODY_NAMES,
        MOTION_REQUIRED_FIELDS,
    )
except ImportError:  # allows the standalone validator to load this file without Isaac Lab
    from contracts import (  # type: ignore[no-redef]
        CONTROL_FREQUENCY_HZ,
        G1_ALL_JOINT_NAMES,
        G1_MOTION_BODY_NAMES,
        MOTION_REQUIRED_FIELDS,
    )


@dataclass(frozen=True)
class ValidatedMotion:
    path: Path
    sha256: str
    fps: float
    source_joint_names: tuple[str, ...]
    source_body_names: tuple[str, ...]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return (self.frame_count - 1) / self.fps

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "source_joint_count": len(self.source_joint_names),
            "source_body_count": len(self.source_body_names),
            "output_joint_names": list(G1_ALL_JOINT_NAMES),
            "output_body_names": list(G1_MOTION_BODY_NAMES),
            "quaternion_convention": "WXYZ",
        }


@dataclass(frozen=True)
class ValidatedMotionDataset:
    """A validated collection stored as contiguous arrays plus clip boundaries."""

    source_path: Path
    paths: tuple[Path, ...]
    clip_sha256: tuple[str, ...]
    sha256: str
    fps: float
    clip_starts: np.ndarray
    clip_lengths: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def path(self) -> Path:
        """Compatibility alias for callers that previously accepted one file."""

        return self.source_path

    @property
    def num_motions(self) -> int:
        return len(self.paths)

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def clip_ends(self) -> np.ndarray:
        """Exclusive global end index of every clip."""

        return self.clip_starts + self.clip_lengths

    @property
    def duration_s(self) -> float:
        return float(np.sum(self.clip_lengths - 1)) / self.fps

    def summary(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "sha256": self.sha256,
            "fps": self.fps,
            "motion_count": self.num_motions,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "output_joint_names": list(G1_ALL_JOINT_NAMES),
            "output_body_names": list(G1_MOTION_BODY_NAMES),
            "quaternion_convention": "WXYZ",
            "clips": [
                {
                    "path": str(path),
                    "sha256": digest,
                    "frame_count": int(length),
                    "duration_s": (int(length) - 1) / self.fps,
                }
                for path, digest, length in zip(self.paths, self.clip_sha256, self.clip_lengths, strict=True)
            ],
        }


def _decode_names(value: np.ndarray, field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{field} must be a 1-D string array; got shape {array.shape}.")
    result = tuple(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in array.tolist())
    if not result or any(not name for name in result):
        raise ValueError(f"{field} must contain non-empty names.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate names.")
    return result


def _order(names: Sequence[str], required: Sequence[str], field: str) -> np.ndarray:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{field} is missing required G1 names: {missing}")
    return np.asarray([names.index(name) for name in required], dtype=np.int64)


def _array(npz: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    value = np.asarray(npz[name], dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_motion(path: str | Path) -> ValidatedMotion:
    """Validate, reorder and return the exact 29-joint/17-body motion input."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".npz":
        raise ValueError(f"Expected one existing G1 .npz motion file, got: {resolved}")

    with np.load(resolved, allow_pickle=False) as npz:
        missing = [name for name in MOTION_REQUIRED_FIELDS if name not in npz]
        if missing:
            raise ValueError(f"Motion NPZ is missing required fields: {missing}")

        fps_values = np.asarray(npz["fps"], dtype=np.float64).reshape(-1)
        if fps_values.size != 1 or not np.isclose(fps_values[0], CONTROL_FREQUENCY_HZ, atol=1.0e-6):
            raise ValueError(
                f"Stage-1 baseline requires {CONTROL_FREQUENCY_HZ:.1f} Hz motion; got {fps_values.tolist()}."
            )
        fps = float(fps_values[0])
        joint_names = _decode_names(npz["joint_names"], "joint_names")
        body_names = _decode_names(npz["body_names"], "body_names")

        if len(joint_names) != len(G1_ALL_JOINT_NAMES) or set(joint_names) != set(G1_ALL_JOINT_NAMES):
            raise ValueError(
                "joint_names must exactly match the 29-joint G1 action/reference contract; "
                f"got {len(joint_names)} names."
            )

        joint_pos_source = _array(npz, "joint_pos")
        joint_vel_source = _array(npz, "joint_vel")
        body_pos_source = _array(npz, "body_pos_w")
        body_quat_source = _array(npz, "body_quat_w")
        body_lin_vel_source = _array(npz, "body_lin_vel_w")
        body_ang_vel_source = _array(npz, "body_ang_vel_w")

    frame_count = int(joint_pos_source.shape[0]) if joint_pos_source.ndim >= 1 else 0
    if frame_count < 2:
        raise ValueError(f"Motion needs at least two frames for t -> t+1 tracking; got {frame_count}.")
    if joint_pos_source.shape != (frame_count, len(joint_names)):
        raise ValueError(f"joint_pos has invalid shape {joint_pos_source.shape}.")
    if joint_vel_source.shape != joint_pos_source.shape:
        raise ValueError(f"joint_vel shape {joint_vel_source.shape} does not match joint_pos.")

    body_count = len(body_names)
    expected_body_shapes = {
        "body_pos_w": (frame_count, body_count, 3),
        "body_quat_w": (frame_count, body_count, 4),
        "body_lin_vel_w": (frame_count, body_count, 3),
        "body_ang_vel_w": (frame_count, body_count, 3),
    }
    actual_body_arrays = {
        "body_pos_w": body_pos_source,
        "body_quat_w": body_quat_source,
        "body_lin_vel_w": body_lin_vel_source,
        "body_ang_vel_w": body_ang_vel_source,
    }
    for name, expected_shape in expected_body_shapes.items():
        if actual_body_arrays[name].shape != expected_shape:
            raise ValueError(f"{name} has shape {actual_body_arrays[name].shape}; expected {expected_shape}.")

    joint_order = _order(joint_names, G1_ALL_JOINT_NAMES, "joint_names")
    body_order = _order(body_names, G1_MOTION_BODY_NAMES, "body_names")

    quaternions = body_quat_source[:, body_order]
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if float(np.max(np.abs(norms - 1.0))) > 1.0e-3:
        raise ValueError("body_quat_w must contain normalized WXYZ quaternions (tolerance 1e-3).")
    quaternions = quaternions / norms

    digest = _sha256_file(resolved)
    return ValidatedMotion(
        path=resolved,
        sha256=digest,
        fps=fps,
        source_joint_names=joint_names,
        source_body_names=body_names,
        joint_pos=np.ascontiguousarray(joint_pos_source[:, joint_order]),
        joint_vel=np.ascontiguousarray(joint_vel_source[:, joint_order]),
        body_pos_w=np.ascontiguousarray(body_pos_source[:, body_order]),
        body_quat_w=np.ascontiguousarray(quaternions),
        body_lin_vel_w=np.ascontiguousarray(body_lin_vel_source[:, body_order]),
        body_ang_vel_w=np.ascontiguousarray(body_ang_vel_source[:, body_order]),
    )


def resolve_motion_files(path: str | Path) -> tuple[Path, tuple[Path, ...]]:
    """Resolve one NPZ, a recursive directory, or a relative-path text manifest."""

    source_path = Path(path).expanduser().resolve()
    if source_path.is_dir():
        paths = tuple(sorted(item.resolve() for item in source_path.rglob("*.npz") if item.is_file()))
    elif source_path.is_file() and source_path.suffix.lower() == ".npz":
        paths = (source_path,)
    elif source_path.is_file() and source_path.suffix.lower() in (".txt", ".lst"):
        entries: list[Path] = []
        for line_number, raw_line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
            entry = raw_line.split("#", maxsplit=1)[0].strip()
            if not entry:
                continue
            item = Path(entry).expanduser()
            if not item.is_absolute():
                item = source_path.parent / item
            item = item.resolve()
            if not item.is_file() or item.suffix.lower() != ".npz":
                raise ValueError(f"Invalid motion manifest entry at line {line_number}: {item}")
            entries.append(item)
        paths = tuple(entries)
    else:
        raise ValueError(
            "Expected a prepared G1 .npz file, a directory containing .npz files, "
            f"or a .txt/.lst manifest; got: {source_path}"
        )

    if not paths:
        raise ValueError(f"No .npz motion files found in: {source_path}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Motion input contains duplicate file paths: {source_path}")
    return source_path, paths


def load_and_validate_motion_dataset(path: str | Path) -> ValidatedMotionDataset:
    """Load a motion collection and concatenate clips without losing boundaries.

    Sampling is intentionally left to the command term. Keeping loading and
    sampling separate makes it impossible for a sampled ``k -> k+1`` pair to
    cross from the end of one clip into the start of another.
    """

    source_path, paths = resolve_motion_files(path)
    motions: list[ValidatedMotion] = []
    for motion_path in paths:
        try:
            motions.append(load_and_validate_motion(motion_path))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid G1 motion clip {motion_path}: {exc}") from exc

    clip_lengths = np.asarray([motion.frame_count for motion in motions], dtype=np.int64)
    clip_starts = np.zeros(len(motions), dtype=np.int64)
    if len(motions) > 1:
        clip_starts[1:] = np.cumsum(clip_lengths[:-1])

    def concatenate(field: str) -> np.ndarray:
        arrays = [getattr(motion, field) for motion in motions]
        if len(arrays) == 1:
            return arrays[0]
        return np.ascontiguousarray(np.concatenate(arrays, axis=0))

    digest = hashlib.sha256()
    for motion in motions:
        digest.update(bytes.fromhex(motion.sha256))

    return ValidatedMotionDataset(
        source_path=source_path,
        paths=paths,
        clip_sha256=tuple(motion.sha256 for motion in motions),
        sha256=digest.hexdigest(),
        fps=motions[0].fps,
        clip_starts=clip_starts,
        clip_lengths=clip_lengths,
        joint_pos=concatenate("joint_pos"),
        joint_vel=concatenate("joint_vel"),
        body_pos_w=concatenate("body_pos_w"),
        body_quat_w=concatenate("body_quat_w"),
        body_lin_vel_w=concatenate("body_lin_vel_w"),
        body_ang_vel_w=concatenate("body_ang_vel_w"),
    )

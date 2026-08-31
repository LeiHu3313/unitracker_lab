#!/usr/bin/env python3
"""Rebuild G1 Teacher body states from repaired 29-DoF prepared motions.

The source clips retain their repaired pelvis pose and 29 joint trajectories.
This tool runs URDF forward kinematics to replace only the body-state arrays
with the current 17-body motion/reward contract.  It deliberately writes to a
new directory, so the source prepared dataset remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
TEACHER_DIR = (
    REPO_ROOT / "source" / "unitracker_lab" / "unitracker_lab" / "tasks" / "manager_based" / "unitracker_teacher"
)
sys.path.insert(0, str(TEACHER_DIR))
from contracts import (  # noqa: E402
    CONTROL_FREQUENCY_HZ,
    G1_ALL_JOINT_NAMES,
    G1_MOTION_BODY_NAMES,
    G1_ROOT_BODY_NAME,
    G1_URDF_SHA256,
)


DEFAULT_URDF = REPO_ROOT / "source" / "unitracker_lab" / "unitracker_lab" / "assets" / "g1" / "g1_29dof_rev_1_0.urdf"
REBUILT_FIELDS = {"joint_names", "body_names", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}


@dataclass(frozen=True)
class UrdfJoint:
    """One URDF tree edge in the parent-link coordinate convention."""

    name: str
    kind: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    value = np.fromstring(text, sep=" ", dtype=np.float64) if text else np.asarray(default, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected a finite 3-vector in URDF, got {text!r}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_urdf(path: Path) -> tuple[UrdfJoint, ...]:
    root = ET.parse(path).getroot()
    links = {element.attrib["name"] for element in root.findall("link")}
    joints: list[UrdfJoint] = []
    child_links: set[str] = set()
    for element in root.findall("joint"):
        kind = element.attrib["type"]
        if kind not in {"fixed", "revolute", "continuous"}:
            raise ValueError(f"Unsupported URDF joint type {kind!r}: {element.attrib['name']}")
        origin = element.find("origin")
        axis_element = element.find("axis")
        axis = _vector(None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0))
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            raise ValueError(f"URDF joint has a zero axis: {element.attrib['name']}")
        joints.append(
            UrdfJoint(
                name=element.attrib["name"],
                kind=kind,
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                origin_xyz=_vector(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0)),
                origin_rotation=Rotation.from_euler(
                    "xyz", _vector(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))
                ).as_matrix(),
                axis=axis / axis_norm,
            )
        )
        child_links.add(element.find("child").attrib["link"])
    roots = links - child_links
    if roots != {G1_ROOT_BODY_NAME}:
        raise ValueError(f"Expected {G1_ROOT_BODY_NAME!r} as unique URDF root, got {sorted(roots)}")
    return tuple(joints)


def _decode_names(value: np.ndarray, field: str) -> tuple[str, ...]:
    names = tuple(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in np.asarray(value).tolist())
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError(f"{field} must be a non-empty, unique one-dimensional name list.")
    return names


def _continuous_quaternion_signs(quat_wxyz: np.ndarray) -> np.ndarray:
    """Choose equivalent quaternion signs continuously along each body track."""

    signs = np.ones(quat_wxyz.shape[:2], dtype=np.float64)
    sign_change = np.sum(quat_wxyz[1:] * quat_wxyz[:-1], axis=-1) < 0.0
    signs[1:] = np.cumprod(np.where(sign_change, -1.0, 1.0), axis=0)
    return quat_wxyz * signs[..., None]


def _forward_kinematics(
    pelvis_pos_w: np.ndarray,
    pelvis_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: tuple[str, ...],
    urdf_joints: tuple[UrdfJoint, ...],
) -> tuple[np.ndarray, np.ndarray]:
    frames = joint_pos.shape[0]
    joint_index = {name: index for index, name in enumerate(joint_names)}
    root_xyzw = pelvis_quat_wxyz[:, (1, 2, 3, 0)]
    root_norm = np.linalg.norm(root_xyzw, axis=-1, keepdims=True)
    if np.any(root_norm < 1.0e-8):
        raise ValueError("Source pelvis quaternion contains a zero norm.")
    positions = {G1_ROOT_BODY_NAME: pelvis_pos_w.astype(np.float64, copy=False)}
    rotations = {G1_ROOT_BODY_NAME: Rotation.from_quat(root_xyzw / root_norm).as_matrix()}

    pending = list(urdf_joints)
    while pending:
        remaining: list[UrdfJoint] = []
        for urdf_joint in pending:
            if urdf_joint.parent not in positions:
                remaining.append(urdf_joint)
                continue
            parent_position = positions[urdf_joint.parent]
            parent_rotation = rotations[urdf_joint.parent]
            positions[urdf_joint.child] = parent_position + np.einsum(
                "fij,j->fi", parent_rotation, urdf_joint.origin_xyz
            )
            origin_rotation_w = parent_rotation @ urdf_joint.origin_rotation
            if urdf_joint.kind == "fixed":
                rotations[urdf_joint.child] = origin_rotation_w
            else:
                if urdf_joint.name not in joint_index:
                    raise ValueError(f"Prepared motion is missing URDF joint {urdf_joint.name!r}.")
                angles = joint_pos[:, joint_index[urdf_joint.name]]
                joint_rotation = Rotation.from_rotvec(angles[:, None] * urdf_joint.axis[None]).as_matrix()
                rotations[urdf_joint.child] = origin_rotation_w @ joint_rotation
        if len(remaining) == len(pending):
            parents = sorted({joint.parent for joint in remaining})
            raise ValueError(f"URDF graph cannot be resolved; missing parent links: {parents}")
        pending = remaining

    missing = [name for name in G1_MOTION_BODY_NAMES if name not in positions]
    if missing:
        raise ValueError(f"URDF does not resolve contract bodies: {missing}")
    body_pos_w = np.stack([positions[name] for name in G1_MOTION_BODY_NAMES], axis=1)
    body_matrix_w = np.stack([rotations[name] for name in G1_MOTION_BODY_NAMES], axis=1)
    body_quat_xyzw = Rotation.from_matrix(body_matrix_w.reshape(-1, 3, 3)).as_quat().reshape(frames, -1, 4)
    body_quat_wxyz = _continuous_quaternion_signs(body_quat_xyzw[..., (3, 0, 1, 2)])
    return body_pos_w.astype(np.float32), body_quat_wxyz.astype(np.float32)


def _angular_velocity(body_quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    frames, bodies = body_quat_wxyz.shape[:2]
    if frames < 2:
        raise ValueError("At least two frames are required to calculate angular velocity.")
    result = np.empty((frames, bodies, 3), dtype=np.float32)
    for body_index in range(bodies):
        rotation = Rotation.from_quat(body_quat_wxyz[:, body_index, (1, 2, 3, 0)])
        interval = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / dt
        result[0, body_index] = interval[0]
        result[-1, body_index] = interval[-1]
        if frames > 2:
            result[1:-1, body_index] = 0.5 * (interval[:-1] + interval[1:])
    return result


def _write_npz_atomic(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **values)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _source_files(input_path: Path) -> tuple[Path, Path, tuple[Path, ...]]:
    resolved = input_path.expanduser().resolve()
    if resolved.is_file() and resolved.suffix.lower() == ".npz":
        return resolved.parent, resolved.parent, (resolved,)
    if resolved.is_dir():
        files = tuple(sorted(path for path in resolved.rglob("*.npz") if path.is_file()))
        if files:
            return resolved, resolved, files
    raise ValueError(f"Expected a prepared .npz file or a directory containing .npz clips: {resolved}")


def rebuild_dataset(input_path: Path, output_dir: Path, urdf: Path, *, overwrite: bool) -> None:
    input_root, relative_root, source_paths = _source_files(input_path)
    output_dir = output_dir.expanduser().resolve()
    urdf = urdf.expanduser().resolve()
    if not urdf.is_file():
        raise ValueError(f"URDF does not exist: {urdf}")
    if _sha256(urdf) != G1_URDF_SHA256:
        raise ValueError(f"URDF hash does not match the current teacher contract: {urdf}")
    if output_dir == input_root or output_dir.is_relative_to(input_root):
        raise ValueError("Output directory must not be the input directory or a child of it.")
    output_paths = tuple(output_dir / source_path.relative_to(relative_root) for source_path in source_paths)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} rebuilt clips already exist under {output_dir}; pass --overwrite to replace them.")

    urdf_joints = _load_urdf(urdf)
    dt = 1.0 / CONTROL_FREQUENCY_HZ
    for clip_number, (source_path, output_path) in enumerate(zip(source_paths, output_paths, strict=True), start=1):
        with np.load(source_path, allow_pickle=False) as source:
            required = {"fps", "joint_names", "body_names", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"}
            missing = sorted(required - set(source.files))
            if missing:
                raise ValueError(f"{source_path} is missing fields required for rebuild: {missing}")
            fps = float(np.asarray(source["fps"], dtype=np.float64).reshape(-1)[0])
            if not np.isclose(fps, CONTROL_FREQUENCY_HZ, atol=1.0e-6):
                raise ValueError(f"{source_path} is {fps:g} Hz; expected {CONTROL_FREQUENCY_HZ:g} Hz.")
            source_joint_names = _decode_names(source["joint_names"], "joint_names")
            source_body_names = _decode_names(source["body_names"], "body_names")
            if len(source_joint_names) != len(G1_ALL_JOINT_NAMES) or set(source_joint_names) != set(G1_ALL_JOINT_NAMES):
                raise ValueError(f"{source_path} does not contain the exact 29-joint G1 set.")
            if G1_ROOT_BODY_NAME not in source_body_names:
                raise ValueError(f"{source_path} does not contain the required pelvis state.")
            joint_pos_source = np.asarray(source["joint_pos"], dtype=np.float32)
            joint_vel_source = np.asarray(source["joint_vel"], dtype=np.float32)
            body_pos_source = np.asarray(source["body_pos_w"], dtype=np.float32)
            body_quat_source = np.asarray(source["body_quat_w"], dtype=np.float32)
            extras = {name: np.asarray(source[name]).copy() for name in source.files if name not in REBUILT_FIELDS}

        frames = joint_pos_source.shape[0]
        if frames < 2 or joint_pos_source.shape != (frames, len(source_joint_names)):
            raise ValueError(f"{source_path} has an invalid joint_pos shape: {joint_pos_source.shape}")
        if joint_vel_source.shape != joint_pos_source.shape:
            raise ValueError(f"{source_path} joint_vel shape does not match joint_pos.")
        if body_pos_source.shape != (frames, len(source_body_names), 3) or body_quat_source.shape != (frames, len(source_body_names), 4):
            raise ValueError(f"{source_path} has invalid source body-state shapes.")
        if not all(np.all(np.isfinite(array)) for array in (joint_pos_source, joint_vel_source, body_pos_source, body_quat_source)):
            raise ValueError(f"{source_path} contains non-finite repaired state values.")

        joint_order = np.asarray([source_joint_names.index(name) for name in G1_ALL_JOINT_NAMES], dtype=np.int64)
        pelvis_index = source_body_names.index(G1_ROOT_BODY_NAME)
        joint_pos = joint_pos_source[:, joint_order]
        joint_vel = joint_vel_source[:, joint_order]
        body_pos_w, body_quat_w = _forward_kinematics(
            body_pos_source[:, pelvis_index], body_quat_source[:, pelvis_index], joint_pos, G1_ALL_JOINT_NAMES, urdf_joints
        )
        edge_order = 2 if frames > 2 else 1
        body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0, edge_order=edge_order).astype(np.float32)
        body_ang_vel_w = _angular_velocity(body_quat_w, dt)
        values = {
            **extras,
            "fps": np.asarray(CONTROL_FREQUENCY_HZ, dtype=np.float64),
            "joint_names": np.asarray(G1_ALL_JOINT_NAMES),
            "body_names": np.asarray(G1_MOTION_BODY_NAMES),
            "joint_pos": np.ascontiguousarray(joint_pos, dtype=np.float32),
            "joint_vel": np.ascontiguousarray(joint_vel, dtype=np.float32),
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "body_state_rebuilt_from": np.asarray(str(source_path)),
            "urdf_sha256": np.asarray(G1_URDF_SHA256),
        }
        _write_npz_atomic(output_path, values)
        print(f"[{clip_number:02d}/{len(source_paths)}] {source_path.name} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="repaired prepared .npz clip or its directory")
    parser.add_argument("output_dir", type=Path, help="new directory for 17-body rebuilt clips")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="G1 29-DoF URDF used for forward kinematics")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output clips")
    args = parser.parse_args()
    rebuild_dataset(args.input, args.output_dir, args.urdf, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

"""Convert a G1 qpos dataset into strict 50-Hz UniTracker Teacher NPZ clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEACHER_DIR = (
    _REPO_ROOT / "source" / "unitracker_lab" / "unitracker_lab" / "tasks" / "manager_based" / "unitracker_teacher"
)
import sys

sys.path.insert(0, str(_TEACHER_DIR))
from contracts import (  # noqa: E402
    CONTROL_FREQUENCY_HZ,
    G1_ALL_JOINT_NAMES,
    G1_CONTROLLED_JOINT_NAMES,
    G1_LOCKED_WRIST_JOINT_NAMES,
    G1_TRACKING_BODY_NAMES,
    G1_URDF_SHA256,
)
from motion_schema import load_and_validate_motion_dataset  # noqa: E402

DEFAULT_URDF = _REPO_ROOT / "source" / "unitracker_lab" / "unitracker_lab" / "assets" / "g1" / "g1_29dof_rev_1_0.urdf"
ROOT_QPOS_NAMES = ("root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz")


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.fromstring(text, sep=" ", dtype=np.float64) if text else np.asarray(default, dtype=np.float64)


def _load_urdf_joints(path: Path) -> tuple[str, tuple[UrdfJoint, ...]]:
    root = ET.parse(path).getroot()
    links = {element.attrib["name"] for element in root.findall("link")}
    joints: list[UrdfJoint] = []
    child_links: set[str] = set()
    for element in root.findall("joint"):
        kind = element.attrib["type"]
        if kind not in ("fixed", "revolute", "continuous"):
            raise ValueError(f"Unsupported URDF joint type {kind!r}: {element.attrib['name']}")
        parent = element.find("parent").attrib["link"]
        child = element.find("child").attrib["link"]
        origin = element.find("origin")
        xyz = _vector(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
        rpy = _vector(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))
        axis_element = element.find("axis")
        axis = _vector(None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0))
        joints.append(
            UrdfJoint(
                name=element.attrib["name"],
                kind=kind,
                parent=parent,
                child=child,
                origin_xyz=xyz,
                origin_rotation=Rotation.from_euler("xyz", rpy).as_matrix(),
                axis=axis / np.linalg.norm(axis),
            )
        )
        child_links.add(child)
    roots = links - child_links
    if roots != {"pelvis"}:
        raise ValueError(f"Expected pelvis as the unique URDF root, got: {sorted(roots)}")
    return "pelvis", tuple(joints)


def _resample(qpos: np.ndarray, source_fps: float) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[1] != len(ROOT_QPOS_NAMES) + 29:
        raise ValueError(f"qpos must have shape [F, 36], got {qpos.shape}")
    if qpos.shape[0] < 2 or not np.all(np.isfinite(qpos)):
        raise ValueError("qpos needs at least two finite frames")

    source_time = np.arange(qpos.shape[0], dtype=np.float64) / source_fps
    frame_count = int(np.floor(source_time[-1] * CONTROL_FREQUENCY_HZ + 1.0e-9)) + 1
    target_time = np.arange(frame_count, dtype=np.float64) / CONTROL_FREQUENCY_HZ

    output = np.empty((frame_count, qpos.shape[1]), dtype=np.float32)
    for column in (*range(3), *range(7, qpos.shape[1])):
        output[:, column] = np.interp(target_time, source_time, qpos[:, column])

    root_xyzw = qpos[:, (4, 5, 6, 3)]
    root_xyzw /= np.linalg.norm(root_xyzw, axis=-1, keepdims=True)
    target_xyzw = Slerp(source_time, Rotation.from_quat(root_xyzw))(target_time).as_quat()
    output[:, 3:7] = target_xyzw[:, (3, 0, 1, 2)]
    return output


def _forward_kinematics(
    qpos: np.ndarray,
    source_joint_names: tuple[str, ...],
    root_link: str,
    joints: tuple[UrdfJoint, ...],
) -> tuple[np.ndarray, np.ndarray]:
    frames = qpos.shape[0]
    source_order = {name: index + len(ROOT_QPOS_NAMES) for index, name in enumerate(source_joint_names)}
    root_quat_xyzw = qpos[:, (4, 5, 6, 3)]
    positions = {root_link: qpos[:, :3].astype(np.float64)}
    rotations = {root_link: Rotation.from_quat(root_quat_xyzw).as_matrix()}

    pending = list(joints)
    while pending:
        remaining: list[UrdfJoint] = []
        for joint in pending:
            if joint.parent not in positions:
                remaining.append(joint)
                continue
            parent_position = positions[joint.parent]
            parent_rotation = rotations[joint.parent]
            positions[joint.child] = parent_position + np.einsum("fij,j->fi", parent_rotation, joint.origin_xyz)
            origin_world = parent_rotation @ joint.origin_rotation
            if joint.kind == "fixed":
                rotations[joint.child] = origin_world
            else:
                if joint.name not in source_order:
                    raise ValueError(f"Source qpos has no URDF joint named {joint.name!r}")
                angle = qpos[:, source_order[joint.name]]
                joint_rotation = Rotation.from_rotvec(angle[:, None] * joint.axis[None]).as_matrix()
                rotations[joint.child] = origin_world @ joint_rotation
        if len(remaining) == len(pending):
            unresolved = sorted({joint.parent for joint in remaining})
            raise ValueError(f"URDF joint graph cannot be resolved; missing parents: {unresolved}")
        pending = remaining

    missing = [name for name in G1_TRACKING_BODY_NAMES if name not in positions]
    if missing:
        raise ValueError(f"URDF does not resolve tracked bodies: {missing}")
    body_pos = np.stack([positions[name] for name in G1_TRACKING_BODY_NAMES], axis=1)
    body_matrix = np.stack([rotations[name] for name in G1_TRACKING_BODY_NAMES], axis=1)
    body_xyzw = Rotation.from_matrix(body_matrix.reshape(-1, 3, 3)).as_quat().reshape(frames, -1, 4)
    body_quat = body_xyzw[..., (3, 0, 1, 2)]

    # Keep equivalent quaternion signs continuous along time for cleaner files.
    sign_change = np.sum(body_quat[1:] * body_quat[:-1], axis=-1) < 0.0
    signs = np.ones((frames, body_quat.shape[1]), dtype=np.float64)
    signs[1:] = np.cumprod(np.where(sign_change, -1.0, 1.0), axis=0)
    body_quat *= signs[..., None]
    return body_pos.astype(np.float32), body_quat.astype(np.float32)


def _angular_velocity(body_quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    frames, bodies = body_quat_wxyz.shape[:2]
    velocity = np.empty((frames, bodies, 3), dtype=np.float32)
    for body_index in range(bodies):
        quat_xyzw = body_quat_wxyz[:, body_index, (1, 2, 3, 0)]
        rotation = Rotation.from_quat(quat_xyzw)
        interval = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / dt
        velocity[0, body_index] = interval[0]
        velocity[-1, body_index] = interval[-1]
        velocity[1:-1, body_index] = 0.5 * (interval[:-1] + interval[1:])
    return velocity


def _write_npz_atomic(path: Path, values: dict[str, object]) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **values)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_dataset(input_dir: Path, output_dir: Path, urdf: Path, *, overwrite: bool) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    urdf = urdf.expanduser().resolve()
    manifest_path = input_dir / "manifest.json"
    motions_dir = input_dir / "motions"
    if not manifest_path.is_file() or not motions_dir.is_dir():
        raise ValueError(f"Expected manifest.json and motions/ under: {input_dir}")
    if _sha256(urdf) != G1_URDF_SHA256:
        raise ValueError(f"URDF hash does not match the Teacher contract: {urdf}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_fps = 1.0 / float(manifest["timestep"])
    source_joint_names = tuple(manifest["qpos_names"][len(ROOT_QPOS_NAMES) :])
    if tuple(manifest["qpos_names"][: len(ROOT_QPOS_NAMES)]) != ROOT_QPOS_NAMES:
        raise ValueError("Source manifest root qpos convention is not xyz + quaternion WXYZ")
    if set(source_joint_names) != set(G1_ALL_JOINT_NAMES):
        raise ValueError("Source manifest does not contain the exact 29-joint G1 set")

    source_paths = tuple(sorted(motions_dir.glob("*.npz")))
    if len(source_paths) != int(manifest["num_motions"]):
        raise ValueError(f"Manifest declares {manifest['num_motions']} motions but found {len(source_paths)} NPZ files")
    output_paths = tuple(output_dir / path.name for path in source_paths)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} outputs already exist under {output_dir}; pass --overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    root_link, urdf_joints = _load_urdf_joints(urdf)
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    wrist_columns = [len(ROOT_QPOS_NAMES) + source_index[name] for name in G1_LOCKED_WRIST_JOINT_NAMES]
    controlled_columns = [len(ROOT_QPOS_NAMES) + source_index[name] for name in G1_CONTROLLED_JOINT_NAMES]
    dt = 1.0 / CONTROL_FREQUENCY_HZ
    clip_records: list[dict[str, object]] = []

    for clip_index, (source_path, output_path) in enumerate(zip(source_paths, output_paths, strict=True), start=1):
        with np.load(source_path, allow_pickle=False) as source:
            if set(source.files) != {"qpos"}:
                raise ValueError(f"Expected only qpos in {source_path}, got fields={source.files}")
            source_qpos = np.asarray(source["qpos"], dtype=np.float64)
        qpos = _resample(source_qpos, source_fps)
        qpos[:, wrist_columns] = 0.0
        body_pos, body_quat = _forward_kinematics(qpos, source_joint_names, root_link, urdf_joints)
        joint_pos = qpos[:, controlled_columns]
        joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=2).astype(np.float32)
        body_lin_vel = np.gradient(body_pos, dt, axis=0, edge_order=2).astype(np.float32)
        body_ang_vel = _angular_velocity(body_quat, dt)

        _write_npz_atomic(
            output_path,
            {
                "fps": np.asarray(CONTROL_FREQUENCY_HZ, dtype=np.float64),
                "joint_names": np.asarray(G1_CONTROLLED_JOINT_NAMES),
                "body_names": np.asarray(G1_TRACKING_BODY_NAMES),
                "joint_pos": joint_pos.astype(np.float32),
                "joint_vel": joint_vel,
                "body_pos_w": body_pos,
                "body_quat_w": body_quat,
                "body_lin_vel_w": body_lin_vel,
                "body_ang_vel_w": body_ang_vel,
                "source_path": np.asarray(str(source_path)),
                "source_fps": np.asarray(source_fps, dtype=np.float64),
                "source_frame_count": np.asarray(source_qpos.shape[0], dtype=np.int64),
                "urdf_sha256": np.asarray(G1_URDF_SHA256),
            },
        )
        clip_records.append(
            {
                "path": output_path.name,
                "source_path": str(source_path),
                "source_frames": int(source_qpos.shape[0]),
                "output_frames": int(qpos.shape[0]),
                "sha256": _sha256(output_path),
            }
        )
        print(f"[{clip_index:02d}/{len(source_paths)}] {source_path.name}: {source_qpos.shape[0]} -> {qpos.shape[0]}")

    (output_dir / "motions.lst").write_text("".join(f"{path.name}\n" for path in output_paths), encoding="utf-8")
    conversion_manifest = {
        "format": "unitracker_g1_teacher_npz_v1",
        "source_dataset": str(input_dir),
        "source_fps": source_fps,
        "output_fps": CONTROL_FREQUENCY_HZ,
        "motion_count": len(output_paths),
        "urdf": str(urdf),
        "urdf_sha256": G1_URDF_SHA256,
        "locked_wrist_joint_names": list(G1_LOCKED_WRIST_JOINT_NAMES),
        "clips": clip_records,
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(conversion_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validated = load_and_validate_motion_dataset(output_dir)
    print(
        f"Converted and validated {validated.num_motions} motions, "
        f"{validated.frame_count} frames at {validated.fps:g} Hz: {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="qpos dataset root containing manifest.json and motions/")
    parser.add_argument("output_dir", type=Path, help="output directory for prepared Teacher NPZ files")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir, args.urdf, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

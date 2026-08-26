"""Ground trusted UFO G1 trajectory pickles against the UniTracker G1 URDF.

The script writes a new qpos dataset compatible with
``convert_g1_qpos_dataset.py``.  It never edits the source pickle files.  The
input pickles are project-generated trusted artifacts; Python pickle must not
be used with untrusted files.

Grounding is contact-aware:

* foot contacts are inferred from collision height, horizontal speed, and tilt;
* optional bounded ankle-only IK levels feet only around inferred support phases;
* a smooth root-Z correction places supporting collision spheres above the plane;
* flight and fall/get-up timing are preserved instead of clamping every frame.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from convert_g1_qpos_dataset import (
    DEFAULT_URDF,
    ROOT_QPOS_NAMES,
    _forward_kinematics,
    _load_urdf_joints,
    _sha256,
)
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEACHER_DIR = (
    _REPO_ROOT / "source" / "unitracker_lab" / "unitracker_lab" / "tasks" / "manager_based" / "unitracker_teacher"
)

sys.path.insert(0, str(_TEACHER_DIR))
from contracts import G1_ALL_JOINT_NAMES, G1_FOOT_BODY_NAMES, G1_TRACKING_BODY_NAMES, G1_URDF_SHA256  # noqa: E402


@dataclass(frozen=True)
class GroundingSettings:
    target_clearance_m: float = 0.002
    contact_height_m: float = 0.015
    contact_speed_mps: float = 0.35
    contact_tilt_deg: float = 8.0
    near_support_margin_m: float = 0.010
    max_root_correction_m: float = 0.05
    max_root_correction_speed_mps: float = 0.5
    max_ankle_correction_rad: float = np.deg2rad(12.0)
    max_ankle_step_rad: float = np.deg2rad(5.0)
    ankle_iterations: int = 4


@dataclass(frozen=True)
class FootCollision:
    centers: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class FootKinematics:
    bottom_z: np.ndarray
    position_xy: np.ndarray
    tilt_deg: np.ndarray
    rotation: np.ndarray


def _load_foot_collisions(urdf: Path) -> dict[str, FootCollision]:
    root = ET.parse(urdf).getroot()
    result: dict[str, FootCollision] = {}
    for foot_name in G1_FOOT_BODY_NAMES:
        link = root.find(f"./link[@name='{foot_name}']")
        if link is None:
            raise ValueError(f"URDF has no foot link {foot_name!r}")
        centers: list[np.ndarray] = []
        radii: list[float] = []
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            if not np.allclose(rpy, 0.0):
                raise ValueError(f"Rotated foot collision geometry is unsupported: {foot_name}")
            sphere = collision.find("./geometry/sphere")
            if sphere is None:
                raise ValueError(f"Expected sphere-only foot collisions in {foot_name}")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            centers.append(xyz)
            radii.append(float(sphere.attrib["radius"]))
        if not centers:
            raise ValueError(f"URDF foot link {foot_name!r} has no collision spheres")
        result[foot_name] = FootCollision(np.stack(centers), np.asarray(radii))
    return result


def _load_joint_limits(urdf: Path, joint_names: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for name in joint_names:
        joint = root.find(f"./joint[@name='{name}']")
        limit = None if joint is None else joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"URDF joint {name!r} has no finite position limits")
        limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return limits


def _foot_kinematics(
    qpos: np.ndarray,
    joint_names: tuple[str, ...],
    root_link: str,
    urdf_joints,
    collisions: dict[str, FootCollision],
) -> FootKinematics:
    body_pos, body_quat = _forward_kinematics(qpos, joint_names, root_link, urdf_joints)
    body_index = {name: index for index, name in enumerate(G1_TRACKING_BODY_NAMES)}
    bottoms: list[np.ndarray] = []
    positions_xy: list[np.ndarray] = []
    tilts: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    for foot_name in G1_FOOT_BODY_NAMES:
        index = body_index[foot_name]
        rotation = Rotation.from_quat(body_quat[:, index, (1, 2, 3, 0)]).as_matrix()
        geometry = collisions[foot_name]
        centers_w = body_pos[:, index, None, :] + np.einsum("fij,pj->fpi", rotation, geometry.centers)
        bottoms.append(np.min(centers_w[..., 2] - geometry.radii[None, :], axis=1))
        positions_xy.append(body_pos[:, index, :2])
        tilts.append(np.degrees(np.arccos(np.clip(rotation[:, 2, 2], -1.0, 1.0))))
        rotations.append(rotation)
    return FootKinematics(
        bottom_z=np.stack(bottoms, axis=1),
        position_xy=np.stack(positions_xy, axis=1),
        tilt_deg=np.stack(tilts, axis=1),
        rotation=np.stack(rotations, axis=1),
    )


def _foot_horizontal_speed(position_xy: np.ndarray, fps: float) -> np.ndarray:
    edge_order = 2 if len(position_xy) >= 3 else 1
    velocity = np.gradient(position_xy, 1.0 / fps, axis=0, edge_order=edge_order)
    return np.linalg.vector_norm(velocity, axis=-1)


def _detect_contacts(kinematics: FootKinematics, fps: float, settings: GroundingSettings) -> np.ndarray:
    speed = _foot_horizontal_speed(kinematics.position_xy, fps)
    support_height = kinematics.bottom_z.min(axis=1, keepdims=True)
    contacts = (
        (kinematics.bottom_z <= settings.contact_height_m)
        & (kinematics.bottom_z <= support_height + settings.near_support_margin_m)
        & (speed <= settings.contact_speed_mps)
        & (kinematics.tilt_deg <= settings.contact_tilt_deg)
    )
    closing_frames = max(2, int(round(0.08 * fps)))
    for foot_index in range(contacts.shape[1]):
        # SciPy's default zero border erodes valid contacts at the beginning and
        # end of a clip. Edge padding preserves a standing first/last frame.
        padding = closing_frames
        padded = np.pad(contacts[:, foot_index], padding, mode="edge")
        padded = binary_closing(padded, structure=np.ones(closing_frames, dtype=bool))
        padded = binary_opening(padded, structure=np.ones(2, dtype=bool))
        contacts[:, foot_index] = padded[padding:-padding]
    return contacts


def _rate_limit(values: np.ndarray, max_step: float, passes: int = 3) -> np.ndarray:
    result = values.copy()
    for _ in range(passes):
        for index in range(1, len(result)):
            result[index] = np.clip(result[index], result[index - 1] - max_step, result[index - 1] + max_step)
        for index in range(len(result) - 2, -1, -1):
            result[index] = np.clip(result[index], result[index + 1] - max_step, result[index + 1] + max_step)
    return result


def _smooth_root_correction(raw: np.ndarray, fps: float, settings: GroundingSettings) -> np.ndarray:
    median_frames = max(3, int(round(0.08 * fps)) | 1)
    correction = median_filter(raw, size=median_frames, mode="nearest")
    correction = gaussian_filter1d(correction, sigma=max(0.5, 0.03 * fps), mode="nearest")
    correction = np.clip(correction, -settings.max_root_correction_m, settings.max_root_correction_m)
    return _rate_limit(correction, settings.max_root_correction_speed_mps / fps)


def _root_height_correction(
    bottom_z: np.ndarray, contacts: np.ndarray, fps: float, settings: GroundingSettings
) -> np.ndarray:
    frame_contact = contacts.any(axis=1)
    if not frame_contact.any():
        return np.zeros(len(bottom_z), dtype=np.float64)
    planted_bottom = np.where(contacts, bottom_z, np.inf).min(axis=1)
    known = np.flatnonzero(frame_contact)
    raw_known = settings.target_clearance_m - planted_bottom[known]
    raw = np.interp(np.arange(len(bottom_z)), known, raw_known)
    return _smooth_root_correction(raw, fps, settings)


def _level_support_feet(
    qpos: np.ndarray,
    joint_names: tuple[str, ...],
    contacts: np.ndarray,
    fps: float,
    settings: GroundingSettings,
    root_link: str,
    urdf_joints,
    collisions: dict[str, FootCollision],
    limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    output = qpos.copy()
    all_delta = np.zeros((len(qpos), 4), dtype=np.float64)
    joint_index = {name: len(ROOT_QPOS_NAMES) + index for index, name in enumerate(joint_names)}
    ankle_names = (
        ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
        ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
    )

    for foot_index, names in enumerate(ankle_names):
        contact_frames = np.flatnonzero(contacts[:, foot_index])
        if contact_frames.size == 0:
            continue
        columns = np.asarray([joint_index[name] for name in names])
        original = qpos[contact_frames].copy()
        working = original.copy()
        for _ in range(settings.ankle_iterations):
            base = _foot_kinematics(working, joint_names, root_link, urdf_joints, collisions)
            residual = base.rotation[:, foot_index, :2, 2]
            jacobian = np.empty((len(working), 2, 2), dtype=np.float64)
            epsilon = 1.0e-4
            for column_index, qpos_column in enumerate(columns):
                perturbed = working.copy()
                perturbed[:, qpos_column] += epsilon
                rotation = _foot_kinematics(perturbed, joint_names, root_link, urdf_joints, collisions).rotation
                jacobian[:, :, column_index] = (rotation[:, foot_index, :2, 2] - residual) / epsilon
            transpose = np.swapaxes(jacobian, 1, 2)
            normal = transpose @ jacobian + 1.0e-6 * np.eye(2)[None]
            rhs = -(transpose @ residual[..., None])[..., 0]
            step = np.linalg.solve(normal, rhs[..., None])[..., 0]
            step = np.clip(step, -settings.max_ankle_step_rad, settings.max_ankle_step_rad)
            for local_column, (name, qpos_column) in enumerate(zip(names, columns, strict=True)):
                lower, upper = limits[name]
                candidate = working[:, qpos_column] + step[:, local_column]
                candidate = np.clip(
                    candidate,
                    original[:, qpos_column] - settings.max_ankle_correction_rad,
                    original[:, qpos_column] + settings.max_ankle_correction_rad,
                )
                working[:, qpos_column] = np.clip(candidate, lower, upper)

        raw_delta = working[:, columns] - original[:, columns]
        ankle_offset = foot_index * 2
        all_delta[contact_frames, ankle_offset : ankle_offset + 2] = raw_delta
        transition_frames = max(1, int(round(0.08 * fps)))
        active = binary_dilation(contacts[:, foot_index], iterations=transition_frames)
        for local_column, qpos_column in enumerate(columns):
            delta = gaussian_filter1d(
                all_delta[:, ankle_offset + local_column], sigma=max(0.5, 0.03 * fps), mode="nearest"
            )
            delta[~active] = 0.0
            output[:, qpos_column] += delta
            all_delta[:, ankle_offset + local_column] = delta
    return output, all_delta


def _quantiles_mm(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    quantiles = np.quantile(values, (0.05, 0.5, 0.95)) * 1000.0
    return {
        "count": int(values.size),
        "min": float(values.min() * 1000.0),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "max": float(values.max() * 1000.0),
    }


def _motion_report(
    name: str,
    source: Path,
    fps: float,
    before: FootKinematics,
    after: FootKinematics,
    contacts: np.ndarray,
    root_correction: np.ndarray,
    ankle_delta: np.ndarray,
) -> dict[str, Any]:
    planted_before = before.bottom_z[contacts]
    planted_after = after.bottom_z[contacts]
    frame_contact = contacts.any(axis=1)
    support_before = np.where(contacts, before.bottom_z, np.inf).min(axis=1)[frame_contact]
    support_after = np.where(contacts, after.bottom_z, np.inf).min(axis=1)[frame_contact]
    tilt_before = before.tilt_deg[contacts]
    tilt_after = after.tilt_deg[contacts]
    return {
        "motion": name,
        "source": str(source),
        "frames": int(len(before.bottom_z)),
        "fps": float(fps),
        "contact_fraction_by_foot": [float(contacts[:, index].mean()) for index in range(contacts.shape[1])],
        "initial_bottom_mm_before": [float(value * 1000.0) for value in before.bottom_z[0]],
        "initial_bottom_mm_after": [float(value * 1000.0) for value in after.bottom_z[0]],
        "initial_tilt_deg_before": [float(value) for value in before.tilt_deg[0]],
        "initial_tilt_deg_after": [float(value) for value in after.tilt_deg[0]],
        "planted_bottom_mm_before": _quantiles_mm(planted_before),
        "planted_bottom_mm_after": _quantiles_mm(planted_after),
        "support_bottom_mm_before": _quantiles_mm(support_before),
        "support_bottom_mm_after": _quantiles_mm(support_after),
        "planted_tilt_deg_before_p95": float(np.quantile(tilt_before, 0.95)) if tilt_before.size else None,
        "planted_tilt_deg_after_p95": float(np.quantile(tilt_after, 0.95)) if tilt_after.size else None,
        "root_z_correction_mm_min": float(root_correction.min() * 1000.0),
        "root_z_correction_mm_max": float(root_correction.max() * 1000.0),
        "root_z_correction_mm_rms": float(np.sqrt(np.mean(np.square(root_correction))) * 1000.0),
        "ankle_correction_deg_abs_max": float(np.degrees(np.max(np.abs(ankle_delta)))),
        "ankle_correction_deg_rms": float(np.degrees(np.sqrt(np.mean(np.square(ankle_delta))))),
    }


def _load_motion(path: Path) -> tuple[str, float, tuple[str, ...], np.ndarray]:
    with path.open("rb") as stream:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated", category=DeprecationWarning)
            blob = pickle.load(stream)  # noqa: S301 - trusted project-generated input only
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: expected a dictionary")
    required = {"joint_q", "dof_names", "sample_rate", "root_quat_format"}
    missing = required.difference(blob)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    joint_names = tuple(str(name) for name in blob["dof_names"])
    if len(joint_names) != len(G1_ALL_JOINT_NAMES) or set(joint_names) != set(G1_ALL_JOINT_NAMES):
        raise ValueError(f"{path}: expected the exact 29-joint G1 set")
    if str(blob["root_quat_format"]).lower() != "xyzw":
        raise ValueError(f"{path}: expected root_quat_format='xyzw'")
    joint_q = np.asarray(blob["joint_q"], dtype=np.float64)
    if joint_q.ndim != 2 or joint_q.shape[1] != len(ROOT_QPOS_NAMES) + len(joint_names):
        raise ValueError(f"{path}: invalid joint_q shape {joint_q.shape}")
    if len(joint_q) < 3 or not np.isfinite(joint_q).all():
        raise ValueError(f"{path}: joint_q must contain at least three finite frames")
    qpos = joint_q.copy()
    qpos[:, 3:7] = joint_q[:, (6, 3, 4, 5)]
    quat_norm = np.linalg.vector_norm(qpos[:, 3:7], axis=1)
    if not np.allclose(quat_norm, 1.0, atol=1.0e-3):
        raise ValueError(f"{path}: root quaternion is not normalized")
    return str(blob.get("name") or path.stem), float(blob["sample_rate"]), joint_names, qpos


def ground_motion(
    path: Path,
    settings: GroundingSettings,
    *,
    level_support_feet: bool,
    root_link: str,
    urdf_joints,
    collisions: dict[str, FootCollision],
    limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    name, fps, joint_names, qpos = _load_motion(path)
    before = _foot_kinematics(qpos, joint_names, root_link, urdf_joints, collisions)
    contacts = _detect_contacts(before, fps, settings)
    if level_support_feet:
        leveled, ankle_delta = _level_support_feet(
            qpos,
            joint_names,
            contacts,
            fps,
            settings,
            root_link,
            urdf_joints,
            collisions,
            limits,
        )
    else:
        leveled = qpos.copy()
        ankle_delta = np.zeros((len(qpos), 4), dtype=np.float64)
    leveled_kinematics = _foot_kinematics(leveled, joint_names, root_link, urdf_joints, collisions)
    root_correction = _root_height_correction(leveled_kinematics.bottom_z, contacts, fps, settings)
    grounded = leveled.copy()
    grounded[:, 2] += root_correction
    after = _foot_kinematics(grounded, joint_names, root_link, urdf_joints, collisions)
    report = _motion_report(name, path, fps, before, after, contacts, root_correction, ankle_delta)
    report["level_support_feet"] = level_support_feet
    report["joint_names"] = list(joint_names)
    return grounded, report


def _write_npz_atomic(path: Path, qpos: np.ndarray) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, qpos=qpos.astype(np.float32))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _aggregate_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    before = np.asarray([value for report in reports for value in report["initial_bottom_mm_before"]])
    after = np.asarray([value for report in reports for value in report["initial_bottom_mm_after"]])
    tilt_before = np.asarray([value for report in reports for value in report["initial_tilt_deg_before"]])
    tilt_after = np.asarray([value for report in reports for value in report["initial_tilt_deg_after"]])
    return {
        "motion_count": len(reports),
        "initial_bottom_mm_before_median": float(np.median(before)),
        "initial_bottom_mm_after_median": float(np.median(after)),
        "initial_bottom_mm_before_min_max": [float(before.min()), float(before.max())],
        "initial_bottom_mm_after_min_max": [float(after.min()), float(after.max())],
        "initial_tilt_deg_before_median_p95": [float(np.median(tilt_before)), float(np.quantile(tilt_before, 0.95))],
        "initial_tilt_deg_after_median_p95": [float(np.median(tilt_after)), float(np.quantile(tilt_after, 0.95))],
    }


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    urdf: Path,
    settings: GroundingSettings,
    *,
    pattern: str,
    level_support_feet: bool,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    urdf = urdf.expanduser().resolve()
    if _sha256(urdf) != G1_URDF_SHA256:
        raise ValueError(f"URDF hash does not match the UniTracker Teacher contract: {urdf}")
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trusted G1 pickle matched {pattern!r} under {input_dir}")
    root_link, urdf_joints = _load_urdf_joints(urdf)
    collisions = _load_foot_collisions(urdf)
    ankle_names = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    limits = _load_joint_limits(urdf, ankle_names)
    motions_dir = output_dir / "motions"
    if not dry_run:
        existing = [motions_dir / f"{path.stem}.npz" for path in paths if (motions_dir / f"{path.stem}.npz").exists()]
        if existing and not overwrite:
            raise FileExistsError(f"{len(existing)} grounded motions already exist; pass --overwrite")
        motions_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    source_fps: float | None = None
    source_joint_names: tuple[str, ...] | None = None
    for index, path in enumerate(paths, start=1):
        grounded, report = ground_motion(
            path,
            settings,
            level_support_feet=level_support_feet,
            root_link=root_link,
            urdf_joints=urdf_joints,
            collisions=collisions,
            limits=limits,
        )
        fps = float(report["fps"])
        joint_names = tuple(report.pop("joint_names"))
        if source_fps is None:
            source_fps = fps
            source_joint_names = joint_names
        if not np.isclose(source_fps, fps) or source_joint_names != joint_names:
            raise ValueError("All source motions must share fps and joint ordering")
        if not dry_run:
            _write_npz_atomic(motions_dir / f"{path.stem}.npz", grounded)
        reports.append(report)
        print(
            f"[{index:02d}/{len(paths)}] {path.name}: "
            f"initial bottom {np.round(report['initial_bottom_mm_before'], 2).tolist()} -> "
            f"{np.round(report['initial_bottom_mm_after'], 2).tolist()} mm; "
            f"tilt {np.round(report['initial_tilt_deg_before'], 2).tolist()} -> "
            f"{np.round(report['initial_tilt_deg_after'], 2).tolist()} deg"
        )

    assert source_fps is not None and source_joint_names is not None
    report_payload = {
        "format": "unitracker_g1_grounding_report_v1",
        "source_dataset": str(input_dir),
        "output_dataset": None if dry_run else str(output_dir),
        "urdf": str(urdf),
        "urdf_sha256": G1_URDF_SHA256,
        "settings": {
            key: (float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in settings.__dict__.items()
        },
        "level_support_feet": level_support_feet,
        "aggregate": _aggregate_report(reports),
        "motions": reports,
    }
    if not dry_run:
        manifest = {
            "format": "unitracker_g1_grounded_qpos_v1",
            "num_motions": len(paths),
            "timestep": 1.0 / source_fps,
            "qpos_names": list(ROOT_QPOS_NAMES) + list(source_joint_names),
            "source_dataset": str(input_dir),
            "grounding_report": "grounding_report.json",
            "urdf": str(urdf),
            "urdf_sha256": G1_URDF_SHA256,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (output_dir / "grounding_report.json").write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report_payload["aggregate"], indent=2, sort_keys=True))
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="trusted UFO directory containing G1 joint_q pickle files")
    parser.add_argument("output_dir", type=Path, help="new qpos dataset directory; source data is never overwritten")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--pattern", default="*.pkl", help="input filename glob, useful for one-motion audits")
    parser.add_argument("--target-clearance-mm", type=float, default=2.0)
    parser.add_argument("--contact-height-mm", type=float, default=15.0)
    parser.add_argument("--contact-speed-mps", type=float, default=0.35)
    parser.add_argument("--contact-tilt-deg", type=float, default=8.0)
    parser.add_argument("--near-support-margin-mm", type=float, default=10.0)
    parser.add_argument("--max-root-correction-mm", type=float, default=50.0)
    parser.add_argument("--max-root-correction-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-ankle-correction-deg", type=float, default=12.0)
    parser.add_argument("--level-support-feet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="analyze and print without creating output files")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = GroundingSettings(
        target_clearance_m=args.target_clearance_mm / 1000.0,
        contact_height_m=args.contact_height_mm / 1000.0,
        contact_speed_mps=args.contact_speed_mps,
        contact_tilt_deg=args.contact_tilt_deg,
        near_support_margin_m=args.near_support_margin_mm / 1000.0,
        max_root_correction_m=args.max_root_correction_mm / 1000.0,
        max_root_correction_speed_mps=args.max_root_correction_speed_mps,
        max_ankle_correction_rad=np.deg2rad(args.max_ankle_correction_deg),
    )
    process_dataset(
        args.input_dir,
        args.output_dir,
        args.urdf,
        settings,
        pattern=args.pattern,
        level_support_feet=args.level_support_feet,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

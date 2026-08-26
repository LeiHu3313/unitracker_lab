from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "rsl_rl" / "teacher"
sys.path.insert(0, str(SCRIPT_DIR))

from ground_g1_pkl_dataset import (  # noqa: E402
    FootKinematics,
    GroundingSettings,
    _detect_contacts,
    _rate_limit,
    _root_height_correction,
)


def test_rate_limit_bounds_both_directions():
    values = np.asarray([0.0, 1.0, -1.0, 0.0])
    limited = _rate_limit(values, max_step=0.2)
    assert np.max(np.abs(np.diff(limited))) <= 0.2 + 1.0e-12


def test_contact_detection_rejects_high_flight_and_keeps_slow_support():
    frames = 12
    bottom = np.full((frames, 2), 0.002)
    bottom[3:9] = 0.20
    xy = np.zeros((frames, 2, 2))
    tilt = np.zeros((frames, 2))
    rotation = np.tile(np.eye(3), (frames, 2, 1, 1))
    contacts = _detect_contacts(
        FootKinematics(bottom_z=bottom, position_xy=xy, tilt_deg=tilt, rotation=rotation),
        fps=50.0,
        settings=GroundingSettings(),
    )
    assert contacts[:3].all()
    assert not contacts[3:9].any()
    assert contacts[9:].all()


def test_root_height_correction_uses_contact_frames_without_clamping_flight():
    bottom = np.asarray([[0.012, 0.20], [0.10, 0.11], [0.008, 0.20]])
    contacts = np.asarray([[True, False], [False, False], [True, False]])
    settings = GroundingSettings(target_clearance_m=0.002, max_root_correction_speed_mps=10.0)
    correction = _root_height_correction(bottom, contacts, fps=50.0, settings=settings)
    np.testing.assert_allclose(correction[[0, 2]], [-0.01, -0.006], atol=0.003)
    assert -0.011 <= correction[1] <= -0.005

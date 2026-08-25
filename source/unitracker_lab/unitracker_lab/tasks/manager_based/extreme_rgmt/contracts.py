"""Hard contracts shared by the standalone Extreme-RGMT task.

Changing an ordered list in this module changes the checkpoint ABI.  Keep the
motion loader, action term, observations, rewards and export code tied to these
constants instead of re-declaring their own order.
"""

from __future__ import annotations

from collections import OrderedDict

G1_PHYSICAL_DOF = 29
G1_CONTROLLED_DOF = 29
G1_TRACKING_BODY_COUNT = 16

G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

G1_WRIST_JOINT_NAMES = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_CONTROLLED_JOINT_NAMES = G1_BODY_JOINT_NAMES + G1_WRIST_JOINT_NAMES
G1_ALL_JOINT_NAMES = G1_CONTROLLED_JOINT_NAMES
G1_DEFAULT_JOINT_POSITIONS = {name: 0.0 for name in G1_ALL_JOINT_NAMES}
for _name in G1_ALL_JOINT_NAMES:
    if "hip_pitch" in _name:
        G1_DEFAULT_JOINT_POSITIONS[_name] = -0.312
    elif "knee" in _name:
        G1_DEFAULT_JOINT_POSITIONS[_name] = 0.669
    elif "ankle_pitch" in _name:
        G1_DEFAULT_JOINT_POSITIONS[_name] = -0.363
    elif "elbow" in _name:
        G1_DEFAULT_JOINT_POSITIONS[_name] = 0.6
G1_DEFAULT_JOINT_POSITIONS.update(
    {
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
)

G1_TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_elbow_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_rubber_hand",
)
G1_ROOT_BODY_NAME = "pelvis"
G1_NON_ROOT_TRACKING_BODY_NAMES = G1_TRACKING_BODY_NAMES[1:]
G1_NON_ROOT_TRACKING_BODY_COUNT = len(G1_NON_ROOT_TRACKING_BODY_NAMES)
G1_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

PHYSICS_DT = 0.002
CONTROL_DECIMATION = 10
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION
CONTROL_FREQUENCY_HZ = 1.0 / CONTROL_DT
ACTION_DIM = G1_CONTROLLED_DOF
PROPRIOCEPTION_DIM = 3 + 3 + 2 * G1_CONTROLLED_DOF
REFERENCE_TOKEN_DIM = 3 + 3 + 3 + G1_CONTROLLED_DOF
HISTORY_LENGTH = 10
REFERENCE_WINDOW_RADIUS = 10
REFERENCE_WINDOW_LENGTH = 2 * REFERENCE_WINDOW_RADIUS + 1

POLICY_OBSERVATION_BLOCK_DIMS = OrderedDict(
    (
        ("proprioception_history", HISTORY_LENGTH * PROPRIOCEPTION_DIM),
        ("previous_action_history", HISTORY_LENGTH * ACTION_DIM),
        ("reference_window", REFERENCE_WINDOW_LENGTH * REFERENCE_TOKEN_DIM),
    )
)
POLICY_OBSERVATION_DIM = sum(POLICY_OBSERVATION_BLOCK_DIMS.values())
CRITIC_PRIVILEGED_DIM = 1 + 3 * G1_TRACKING_BODY_COUNT + 6 * G1_TRACKING_BODY_COUNT + 3
CRITIC_OBSERVATION_DIM = POLICY_OBSERVATION_DIM + CRITIC_PRIVILEGED_DIM

MOTION_REQUIRED_FIELDS = (
    "fps",
    "joint_names",
    "body_names",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
MOTION_QUATERNION_CONVENTION = "WXYZ"
G1_URDF_SHA256 = "8df048597b758a4f868c1eef12ba995e331420a5aceef810a07c12e3b208ac13"

TRACKING_REWARD_SPECS = OrderedDict(
    (
        ("anchor_orientation", {"weight": 0.5, "sigma": 0.40}),
        ("body_position", {"weight": 1.0, "sigma": 0.30}),
        ("body_orientation", {"weight": 1.0, "sigma": 0.40}),
        ("body_linear_velocity", {"weight": 1.0, "sigma": 1.00}),
        ("body_angular_velocity", {"weight": 1.0, "sigma": 2.50}),
    )
)
REGULARIZATION_REWARD_WEIGHTS = {
    "action_rate": -0.1,
    "joint_position_limits": -10.0,
    "undesired_contacts": -0.1,
    "foot_slip": -0.1,
}
FOOT_CONTACT_FORCE_THRESHOLD_N = 1.0
TERMINATION_SPECS = {
    "projected_gravity": {"threshold": 0.8},
    "pelvis_position": {"threshold": 0.4},
}

ASSET_DR_RANGES = {
    "ground_friction": (0.1, 1.75),
    "added_base_mass_kg": (-3.0, 6.0),
    "base_com_x_m": (-0.025, 0.025),
    "base_com_yz_m": (-0.05, 0.05),
    "motor_strength_scale": (0.8, 1.2),
    "pd_gain_scale": (0.8, 1.2),
    "motor_zero_offset_rad": (-0.01, 0.01),
    "joint_armature_scale": (1.0, 1.05),
    "external_push_interval_s": (1.0, 3.0),
}

def policy_observation_slices() -> dict[str, tuple[int, int]]:
    """Return stable half-open offsets for every policy-observation block."""

    result: dict[str, tuple[int, int]] = {}
    start = 0
    for name, width in POLICY_OBSERVATION_BLOCK_DIMS.items():
        result[name] = (start, start + width)
        start += width
    return result


def reference_promotion_mask(just_reset, episode_length):
    """Return which envs may promote their reference phase after this step.

    NumPy arrays and torch tensors both implement the operators used here. A
    reset performed inside ``env.step`` has episode length zero and must retain
    k -> k+1 for the observation returned from that step.
    """

    return ~(just_reset & (episode_length == 0))


def contract_dict() -> dict[str, object]:
    """JSON-serializable contract snapshot stored next to each training run."""

    return {
        "robot": "unitree_g1_29dof_rev_1_0",
        "physical_dof": G1_PHYSICAL_DOF,
        "controlled_dof": G1_CONTROLLED_DOF,
        "controlled_joint_names": list(G1_CONTROLLED_JOINT_NAMES),
        "wrist_joint_names": list(G1_WRIST_JOINT_NAMES),
        "default_joint_positions": G1_DEFAULT_JOINT_POSITIONS,
        "tracking_body_names": list(G1_TRACKING_BODY_NAMES),
        "non_root_tracking_body_names": list(G1_NON_ROOT_TRACKING_BODY_NAMES),
        "root_body_name": G1_ROOT_BODY_NAME,
        "foot_body_names": list(G1_FOOT_BODY_NAMES),
        "physics_dt": PHYSICS_DT,
        "control_decimation": CONTROL_DECIMATION,
        "control_dt": CONTROL_DT,
        "action_dim": ACTION_DIM,
        "proprioception_dim": PROPRIOCEPTION_DIM,
        "reference_token_dim": REFERENCE_TOKEN_DIM,
        "history_length": HISTORY_LENGTH,
        "reference_window_radius": REFERENCE_WINDOW_RADIUS,
        "reference_window_length": REFERENCE_WINDOW_LENGTH,
        "actor_observation_dim": POLICY_OBSERVATION_DIM,
        "critic_observation_dim": CRITIC_OBSERVATION_DIM,
        "observation_blocks": {
            name: {"start": bounds[0], "end": bounds[1], "dim": bounds[1] - bounds[0]}
            for name, bounds in policy_observation_slices().items()
        },
        "motion_schema": {
            "required_fields": list(MOTION_REQUIRED_FIELDS),
            "fps": CONTROL_FREQUENCY_HZ,
            "quaternion_convention": MOTION_QUATERNION_CONVENTION,
        },
        "urdf_sha256": G1_URDF_SHA256,
        "tracking_reward_specs": TRACKING_REWARD_SPECS,
        "regularization_reward_weights": REGULARIZATION_REWARD_WEIGHTS,
        "termination_specs": TERMINATION_SPECS,
        "asset_dr_ranges": ASSET_DR_RANGES,
    }


assert len(G1_CONTROLLED_JOINT_NAMES) == G1_CONTROLLED_DOF
assert len(set(G1_ALL_JOINT_NAMES)) == G1_PHYSICAL_DOF
assert len(G1_TRACKING_BODY_NAMES) == G1_TRACKING_BODY_COUNT
assert G1_TRACKING_BODY_NAMES[0] == G1_ROOT_BODY_NAME
assert len(G1_NON_ROOT_TRACKING_BODY_NAMES) == G1_NON_ROOT_TRACKING_BODY_COUNT == 15
assert PROPRIOCEPTION_DIM == 64
assert REFERENCE_TOKEN_DIM == 38
assert POLICY_OBSERVATION_DIM == 1728
assert CRITIC_OBSERVATION_DIM == 1876

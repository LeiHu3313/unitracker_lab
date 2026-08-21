"""Hard contracts shared by every G1 Stage-1 teacher component.

Changing an ordered list in this module changes the checkpoint ABI.  Keep the
motion loader, action term, observations, rewards and export code tied to these
constants instead of re-declaring their own order.
"""

from __future__ import annotations

from collections import OrderedDict

G1_PHYSICAL_DOF = 29
G1_CONTROLLED_DOF = 23
G1_LOCKED_WRIST_DOF = 6
G1_TRACKING_BODY_COUNT = 16

G1_CONTROLLED_JOINT_NAMES = (
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

G1_LOCKED_WRIST_JOINT_NAMES = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_LOCKED_WRIST_POSITIONS = (0.0,) * G1_LOCKED_WRIST_DOF

G1_ALL_JOINT_NAMES = G1_CONTROLLED_JOINT_NAMES + G1_LOCKED_WRIST_JOINT_NAMES
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
G1_HAND_BODY_NAMES = ("left_rubber_hand", "right_rubber_hand")
# The paper's local five-point term: trunk, both ankle endpoints, both wrists.
# Keep this order stable because it is recorded in each run's contract snapshot.
G1_LOCAL_FIVE_POINT_BODY_NAMES = ("torso_link", *G1_FOOT_BODY_NAMES, *G1_HAND_BODY_NAMES)
G1_TERMINATION_KEY_BODY_NAMES = (G1_ROOT_BODY_NAME, *G1_FOOT_BODY_NAMES, *G1_HAND_BODY_NAMES)

PHYSICS_DT = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION
CONTROL_FREQUENCY_HZ = 1.0 / CONTROL_DT
ACTION_DIM = G1_CONTROLLED_DOF

# The insertion order is the tensor concatenation order in observations.py.
ORACLE_OBSERVATION_BLOCK_DIMS = OrderedDict(
    (
        ("current_root_height", 1),
        ("current_projected_gravity", 3),
        ("current_root_lin_vel_local", 3),
        ("current_root_ang_vel_local", 3),
        ("current_non_root_body_pos_local", 45),
        ("current_non_root_body_ori_rot6d_local", 90),
        ("current_non_root_body_lin_vel_local", 45),
        ("current_non_root_body_ang_vel_local", 45),
        ("current_joint_pos", 23),
        ("current_joint_vel", 23),
        ("previous_action", 23),
        ("next_root_height_error", 1),
        ("next_root_ori_error_rot6d", 6),
        ("next_root_lin_vel_error_local", 3),
        ("next_root_ang_vel_error_local", 3),
        ("next_non_root_body_pos_error_local", 45),
        ("next_non_root_body_ori_error_rot6d", 90),
        ("next_non_root_body_lin_vel_error_local", 45),
        ("next_non_root_body_ang_vel_error_local", 45),
        ("next_joint_pos_error", 23),
        ("next_joint_vel_error", 23),
    )
)
ORACLE_OBSERVATION_DIM = sum(ORACLE_OBSERVATION_BLOCK_DIMS.values())
CRITIC_OBSERVATION_DIM = ORACLE_OBSERVATION_DIM

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
MOTION_WRIST_POSITION_TOLERANCE_RAD = 0.05
MOTION_WRIST_VELOCITY_TOLERANCE_RAD_S = 0.1
G1_URDF_SHA256 = "8df048597b758a4f868c1eef12ba995e331420a5aceef810a07c12e3b208ac13"

TRACKING_REWARD_SPECS = OrderedDict(
    (
        # Global pelvis/base anchor.
        ("base_position", {"weight": 1.0, "sigma": 0.30}),
        ("base_orientation", {"weight": 1.0, "sigma": 0.40}),
        # Relative whole-body pose.
        ("local_five_point_position", {"weight": 2.0, "sigma": 0.12}),
        ("body_position", {"weight": 1.0, "sigma": 0.30}),
        ("body_orientation", {"weight": 1.0, "sigma": 0.40}),
        # Joint and whole-body dynamical tracking.
        ("joint_position", {"weight": 0.75, "sigma": 0.30}),
        ("joint_velocity", {"weight": 0.5, "sigma": 1.0}),
        ("body_linear_velocity", {"weight": 1.0, "sigma": 1.0}),
        ("body_angular_velocity", {"weight": 1.0, "sigma": 3.14}),
        # Extra global torso rotation signal for high-dynamic motions.
        ("torso_orientation", {"weight": 1.0, "sigma": 0.40}),
    )
)
REGULARIZATION_REWARD_WEIGHTS = {
    "action_rate": -0.1,
    "controlled_joint_torque": -1.0e-6,
    "foot_slip": -1.0,
    "early_termination": -100.0,
}
TERMINATION_SPECS = {
    "projected_gravity": {"threshold": 0.8},
    "key_body_height": {"threshold": 0.4},
    "pelvis_position": {"threshold": 0.4},
}

REWARD_CURRICULUM = None
ASSET_DR_RANGES = {
    "static_friction": (0.3, 1.6),
    "dynamic_friction": (0.3, 1.2),
    "restitution": (0.0, 0.5),
    "torso_pelvis_com_x": (-0.03, 0.03),
    "torso_pelvis_com_yz": (-0.05, 0.05),
    "link_mass_scale": (0.8, 1.2),
}


def oracle_observation_slices() -> dict[str, tuple[int, int]]:
    """Return the stable half-open offsets for every oracle observation block."""

    result: dict[str, tuple[int, int]] = {}
    start = 0
    for name, width in ORACLE_OBSERVATION_BLOCK_DIMS.items():
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
        "locked_wrist_dof": G1_LOCKED_WRIST_DOF,
        "controlled_joint_names": list(G1_CONTROLLED_JOINT_NAMES),
        "locked_wrist_joint_names": list(G1_LOCKED_WRIST_JOINT_NAMES),
        "locked_wrist_positions": list(G1_LOCKED_WRIST_POSITIONS),
        "default_joint_positions": G1_DEFAULT_JOINT_POSITIONS,
        "tracking_body_names": list(G1_TRACKING_BODY_NAMES),
        "non_root_tracking_body_names": list(G1_NON_ROOT_TRACKING_BODY_NAMES),
        "root_body_name": G1_ROOT_BODY_NAME,
        "foot_body_names": list(G1_FOOT_BODY_NAMES),
        "local_five_point_body_names": list(G1_LOCAL_FIVE_POINT_BODY_NAMES),
        "physics_dt": PHYSICS_DT,
        "control_decimation": CONTROL_DECIMATION,
        "control_dt": CONTROL_DT,
        "action_dim": ACTION_DIM,
        "actor_observation_dim": ORACLE_OBSERVATION_DIM,
        "critic_observation_dim": CRITIC_OBSERVATION_DIM,
        "observation_blocks": {
            name: {"start": bounds[0], "end": bounds[1], "dim": bounds[1] - bounds[0]}
            for name, bounds in oracle_observation_slices().items()
        },
        "motion_schema": {
            "required_fields": list(MOTION_REQUIRED_FIELDS),
            "fps": CONTROL_FREQUENCY_HZ,
            "quaternion_convention": MOTION_QUATERNION_CONVENTION,
        },
        "urdf_sha256": G1_URDF_SHA256,
        "tracking_reward_specs": TRACKING_REWARD_SPECS,
        "regularization_reward_weights": REGULARIZATION_REWARD_WEIGHTS,
        "reward_curriculum": REWARD_CURRICULUM,
        "termination_key_body_names": list(G1_TERMINATION_KEY_BODY_NAMES),
        "termination_specs": TERMINATION_SPECS,
        "asset_dr_ranges": ASSET_DR_RANGES,
    }


assert len(G1_CONTROLLED_JOINT_NAMES) == G1_CONTROLLED_DOF
assert len(G1_LOCKED_WRIST_JOINT_NAMES) == G1_LOCKED_WRIST_DOF
assert len(set(G1_ALL_JOINT_NAMES)) == G1_PHYSICAL_DOF
assert len(G1_TRACKING_BODY_NAMES) == G1_TRACKING_BODY_COUNT
assert G1_TRACKING_BODY_NAMES[0] == G1_ROOT_BODY_NAME
assert len(G1_NON_ROOT_TRACKING_BODY_NAMES) == G1_NON_ROOT_TRACKING_BODY_COUNT == 15
assert len(G1_LOCAL_FIVE_POINT_BODY_NAMES) == 5
assert set(G1_LOCAL_FIVE_POINT_BODY_NAMES).issubset(G1_TRACKING_BODY_NAMES)
assert len(G1_TERMINATION_KEY_BODY_NAMES) == 5
assert set(G1_TERMINATION_KEY_BODY_NAMES).issubset(G1_TRACKING_BODY_NAMES)
assert ORACLE_OBSERVATION_DIM == 588

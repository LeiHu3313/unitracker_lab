"""Hard contracts shared by every G1 Stage-1 teacher component.

Changing an ordered list in this module changes the checkpoint ABI.  Keep the
motion loader, action term, observations, rewards and export code tied to these
constants instead of re-declaring their own order.
"""

from __future__ import annotations

from collections import OrderedDict

G1_PHYSICAL_DOF = 29
G1_WRIST_DOF = 6

# The base body was the former 23-DoF action contract.  Keep its order as the
# prefix of the 29-DoF contract so that only the six wrist coordinates are
# appended during the migration.
G1_BASE_JOINT_NAMES = (
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
G1_ALL_JOINT_NAMES = G1_BASE_JOINT_NAMES + G1_WRIST_JOINT_NAMES
# All physical joints are now reference-tracked and controlled.  This alias is
# intentionally retained for manager/reward code that consumes the action ABI.
G1_CONTROLLED_JOINT_NAMES = G1_ALL_JOINT_NAMES
G1_CONTROLLED_DOF = G1_PHYSICAL_DOF
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

# Motion data retains waist and palm states, while actor observations and the
# MimicLite-aligned whole-body pose reward use the 14-body subset below.
G1_MOTION_BODY_NAMES = (
    "pelvis",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
)
G1_OBSERVATION_BODY_NAMES = (
    "pelvis",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
# Same semantic list as MimicLite's tracking bodies, with ankle-roll replacing
# its unavailable toe link.  Pelvis stays in the list: its root-local pose
# error is identically zero, as it is in MimicLite's averaging convention.
G1_REWARD_BODY_NAMES = G1_OBSERVATION_BODY_NAMES
# MimicLite does not directly reward ankle q/dq.  The ankle remains controlled
# and reference-conditioned, but is shaped through ankle-roll endpoint pose,
# orientation, velocity and contact rewards instead.
G1_REWARD_JOINT_NAMES = tuple(name for name in G1_ALL_JOINT_NAMES if "ankle_" not in name)
G1_ROOT_BODY_NAME = "pelvis"
G1_MOTION_BODY_COUNT = len(G1_MOTION_BODY_NAMES)
G1_OBSERVATION_BODY_COUNT = len(G1_OBSERVATION_BODY_NAMES)
G1_NON_ROOT_MOTION_BODY_NAMES = G1_MOTION_BODY_NAMES[1:]
G1_NON_ROOT_MOTION_BODY_COUNT = len(G1_NON_ROOT_MOTION_BODY_NAMES)
G1_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_HAND_BODY_NAMES = ("left_rubber_hand", "right_rubber_hand")
# The paper's local five-point term: trunk, both ankle endpoints, both wrists.
# Keep this order stable because it is recorded in each run's contract snapshot.
G1_LOCAL_FIVE_POINT_BODY_NAMES = ("torso_link", *G1_FOOT_BODY_NAMES, *G1_HAND_BODY_NAMES)

PHYSICS_DT = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION
CONTROL_FREQUENCY_HZ = 1.0 / CONTROL_DT
ACTION_DIM = G1_CONTROLLED_DOF

# MimicLite-style temporal layout.  The ordering within each tuple is the
# concatenation order, with index zero always denoting the current control
# step.  Reference offsets are clipped at each motion clip boundary.
TEACHER_STATE_HISTORY_OFFSETS = (0, 1, 2, 3, 4, 8, 16)
TEACHER_ACTION_HISTORY_OFFSETS = (0, 1, 2)
TEACHER_REFERENCE_OFFSETS = (-8, -4, -2, 0, 1, 2, 3, 4)
TEACHER_TRACKING_FEEDBACK_OFFSETS = (0, 1)

# The two groups are deliberately separate in the environment output but are
# concatenated in this exact order for both PPO actor and critic.  The teacher
# is fully privileged: no observation noise is injected and actor/critic see
# the same 1425-D vector.
TEACHER_STATE_OBSERVATION_BLOCK_DIMS = OrderedDict(
    (
        ("root_angular_velocity_history_b", len(TEACHER_STATE_HISTORY_OFFSETS) * 3),
        ("projected_gravity_history_b", len(TEACHER_STATE_HISTORY_OFFSETS) * 3),
        ("joint_position_history_rel_default", len(TEACHER_STATE_HISTORY_OFFSETS) * G1_PHYSICAL_DOF),
        ("joint_velocity_history", len(TEACHER_STATE_HISTORY_OFFSETS) * G1_PHYSICAL_DOF),
        ("previous_action_history", len(TEACHER_ACTION_HISTORY_OFFSETS) * ACTION_DIM),
        # Current full-privileged body state.  Position origin is the pelvis'
        # ground projection; axes are the complete current pelvis frame.
        ("body_position_b", G1_OBSERVATION_BODY_COUNT * 3),
        ("body_linear_velocity_b", G1_OBSERVATION_BODY_COUNT * 3),
        # The action in policy units after action delay/filter.  The current
        # JointPositionAction has neither, so it equals the raw policy action.
        ("applied_action", ACTION_DIM),
        ("applied_joint_torque", G1_PHYSICAL_DOF),
    )
)
TEACHER_REFERENCE_OBSERVATION_BLOCK_DIMS = OrderedDict(
    (
        # Target-only root trajectory in the reference current pelvis-ground,
        # yaw-only frame, followed by root orientation in the robot full frame.
        ("reference_root_position_trajectory_yaw_local", len(TEACHER_REFERENCE_OFFSETS) * 3),
        ("reference_root_orientation_trajectory_robot_b", len(TEACHER_REFERENCE_OFFSETS) * 6),
        ("reference_joint_position_trajectory", len(TEACHER_REFERENCE_OFFSETS) * G1_PHYSICAL_DOF),
        # Direct global/root tracking feedback, expressed in current robot full
        # pelvis axes.  This is privileged teacher information.
        ("reference_root_position_trajectory_robot_b", len(TEACHER_REFERENCE_OFFSETS) * 3),
        # At ref t and t+1, robot and reference poses use their respective
        # pelvis-ground, yaw-only frames; velocity errors remain world-frame.
        ("body_position_error_yaw_local", len(TEACHER_TRACKING_FEEDBACK_OFFSETS) * G1_OBSERVATION_BODY_COUNT * 3),
        ("body_orientation_error_yaw_local_rot6d", len(TEACHER_TRACKING_FEEDBACK_OFFSETS) * G1_OBSERVATION_BODY_COUNT * 6),
        ("body_linear_velocity_error_w", len(TEACHER_TRACKING_FEEDBACK_OFFSETS) * G1_OBSERVATION_BODY_COUNT * 3),
        ("body_angular_velocity_error_w", len(TEACHER_TRACKING_FEEDBACK_OFFSETS) * G1_OBSERVATION_BODY_COUNT * 3),
    )
)
TEACHER_STATE_OBSERVATION_DIM = sum(TEACHER_STATE_OBSERVATION_BLOCK_DIMS.values())
TEACHER_REFERENCE_OBSERVATION_DIM = sum(TEACHER_REFERENCE_OBSERVATION_BLOCK_DIMS.values())
TEACHER_OBSERVATION_DIM = TEACHER_STATE_OBSERVATION_DIM + TEACHER_REFERENCE_OBSERVATION_DIM
CRITIC_OBSERVATION_DIM = TEACHER_OBSERVATION_DIM

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
        # Teacher tracking objective.  The adaptive sampler uses these sigmas
        # to keep its difficulty score on the same relative scale.
        ("torso_position", {"weight": 0.5, "sigma": 0.30}),
        ("torso_orientation", {"weight": 0.5, "sigma": 0.40}),
        ("torso_linear_velocity", {"weight": 0.5, "sigma": 1.00}),
        ("torso_angular_velocity", {"weight": 0.5, "sigma": 2.50}),
        # Re-emphasize task-space endpoints that can otherwise be diluted by
        # averaging the whole-body errors over all 15 non-root tracking links.
        ("local_five_point_position", {"weight": 0.5, "sigma": 0.12}),
        ("local_foot_orientation", {"weight": 0.1, "sigma": 0.30}),
        # Relative whole-body pose is the primary tracking objective.
        ("body_position", {"weight": 1.0, "sigma": 0.30}),
        ("body_orientation", {"weight": 1.0, "sigma": 0.40}),
        # Joint and whole-body dynamical tracking.
        ("joint_position", {"weight": 0.5, "sigma": 0.25}),
        ("joint_velocity", {"weight": 0.5, "sigma": 2.50}),
        ("body_linear_velocity", {"weight": 0.5, "sigma": 1.00}),
        ("body_angular_velocity", {"weight": 0.5, "sigma": 2.50}),
    )
)
REGULARIZATION_REWARD_WEIGHTS = {
    "action_rate": -0.01,
    "controlled_joint_velocity": -1.0e-4,
    "controlled_joint_position_limits": -10.0,
    "foot_slip": -1.5,
    "early_termination": -50.0,
}
FOOT_CONTACT_FORCE_THRESHOLD_N = 1.0
TERMINATION_SPECS = {
    "projected_gravity": {"threshold": 0.8},
    "pelvis_position": {"threshold": 0.4},
}

REWARD_CURRICULUM = None
ASSET_DR_RANGES = {
    "static_friction": (0.8, 1.2),
    "dynamic_friction": (0.8, 1.2),
    "restitution": (0.0, 0.15),
    "torso_pelvis_com_x": (-0.01, 0.01),
    "torso_pelvis_com_yz": (-0.01, 0.01),
    "link_mass_scale": (0.95, 1.05),
}
PUSH_EVENT_SPECS = {
    # A deliberately mild planar velocity kick.  It is less frequent and
    # smaller than the BeyondMimic baseline while foot tracking is stabilizing.
    "interval_range_s": (4.0, 8.0),
    "velocity_range": {
        "x": (-0.15, 0.15),
        "y": (-0.15, 0.15),
    },
}


def observation_slices(block_dims: OrderedDict[str, int]) -> dict[str, tuple[int, int]]:
    """Return the stable half-open offsets for one ordered observation group."""

    result: dict[str, tuple[int, int]] = {}
    start = 0
    for name, width in block_dims.items():
        result[name] = (start, start + width)
        start += width
    return result


def teacher_observation_slices() -> dict[str, dict[str, tuple[int, int]]]:
    """Return per-group offsets for the fully privileged teacher observation."""

    return {
        "teacher_state": observation_slices(TEACHER_STATE_OBSERVATION_BLOCK_DIMS),
        "teacher_reference": observation_slices(TEACHER_REFERENCE_OBSERVATION_BLOCK_DIMS),
    }


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
        "wrist_dof": G1_WRIST_DOF,
        "controlled_joint_names": list(G1_CONTROLLED_JOINT_NAMES),
        "wrist_joint_names": list(G1_WRIST_JOINT_NAMES),
        "default_joint_positions": G1_DEFAULT_JOINT_POSITIONS,
        "motion_body_names": list(G1_MOTION_BODY_NAMES),
        "observation_body_names": list(G1_OBSERVATION_BODY_NAMES),
        "reward_body_names": list(G1_REWARD_BODY_NAMES),
        "reward_joint_names": list(G1_REWARD_JOINT_NAMES),
        "non_root_motion_body_names": list(G1_NON_ROOT_MOTION_BODY_NAMES),
        "root_body_name": G1_ROOT_BODY_NAME,
        "foot_body_names": list(G1_FOOT_BODY_NAMES),
        "local_five_point_body_names": list(G1_LOCAL_FIVE_POINT_BODY_NAMES),
        "physics_dt": PHYSICS_DT,
        "control_decimation": CONTROL_DECIMATION,
        "control_dt": CONTROL_DT,
        "action_dim": ACTION_DIM,
        "teacher_state_history_offsets": list(TEACHER_STATE_HISTORY_OFFSETS),
        "teacher_action_history_offsets": list(TEACHER_ACTION_HISTORY_OFFSETS),
        "teacher_reference_offsets": list(TEACHER_REFERENCE_OFFSETS),
        "teacher_tracking_feedback_offsets": list(TEACHER_TRACKING_FEEDBACK_OFFSETS),
        "actor_observation_dim": TEACHER_OBSERVATION_DIM,
        "critic_observation_dim": CRITIC_OBSERVATION_DIM,
        "observation_groups": {
            group_name: {
                "dim": sum(
                    width
                    for width in (
                        TEACHER_STATE_OBSERVATION_BLOCK_DIMS
                        if group_name == "teacher_state"
                        else TEACHER_REFERENCE_OBSERVATION_BLOCK_DIMS
                    ).values()
                ),
                "blocks": {
                    name: {"start": bounds[0], "end": bounds[1], "dim": bounds[1] - bounds[0]}
                    for name, bounds in group_slices.items()
                },
            }
            for group_name, group_slices in teacher_observation_slices().items()
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
        "termination_specs": TERMINATION_SPECS,
        "asset_dr_ranges": ASSET_DR_RANGES,
        "push_event_specs": PUSH_EVENT_SPECS,
    }


assert len(G1_CONTROLLED_JOINT_NAMES) == G1_CONTROLLED_DOF
assert len(G1_WRIST_JOINT_NAMES) == G1_WRIST_DOF
assert len(set(G1_ALL_JOINT_NAMES)) == G1_PHYSICAL_DOF
assert G1_MOTION_BODY_NAMES[0] == G1_ROOT_BODY_NAME
assert G1_OBSERVATION_BODY_NAMES[0] == G1_ROOT_BODY_NAME
assert len(G1_MOTION_BODY_NAMES) == G1_MOTION_BODY_COUNT == 17
assert len(G1_OBSERVATION_BODY_NAMES) == G1_OBSERVATION_BODY_COUNT == 14
assert set(G1_OBSERVATION_BODY_NAMES).issubset(G1_MOTION_BODY_NAMES)
assert len(G1_REWARD_BODY_NAMES) == 14
assert set(G1_REWARD_BODY_NAMES).issubset(G1_MOTION_BODY_NAMES)
assert len(G1_REWARD_JOINT_NAMES) == 25
assert set(G1_REWARD_JOINT_NAMES).issubset(G1_ALL_JOINT_NAMES)
assert len(G1_NON_ROOT_MOTION_BODY_NAMES) == G1_NON_ROOT_MOTION_BODY_COUNT == 16
assert len(G1_LOCAL_FIVE_POINT_BODY_NAMES) == 5
assert set(G1_LOCAL_FIVE_POINT_BODY_NAMES).issubset(G1_MOTION_BODY_NAMES)
assert TEACHER_STATE_OBSERVATION_DIM == 677
assert TEACHER_REFERENCE_OBSERVATION_DIM == 748
assert TEACHER_OBSERVATION_DIM == CRITIC_OBSERVATION_DIM == 1425

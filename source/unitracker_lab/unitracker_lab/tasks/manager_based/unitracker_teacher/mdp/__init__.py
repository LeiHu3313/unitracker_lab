"""MDP terms for the G1-only UniTracker Stage-1 teacher."""

from isaaclab.envs.mdp import (
    JointPositionActionCfg,
    push_by_setting_velocity,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    time_out,
)

from .commands import MotionCommand, MotionCommandCfg
from .curriculum import linear_curriculum_scale, regularization_scale
from .observations import teacher_reference_observation, teacher_state_observation
from .rewards import (
    action_rate_curriculum,
    action_rate_penalty,
    body_angular_velocity_tracking_exp,
    body_linear_velocity_tracking_exp,
    body_orientation_tracking_exp,
    body_position_tracking_exp,
    controlled_joint_position_limit_penalty,
    controlled_joint_velocity_penalty,
    early_termination_penalty,
    early_termination_penalty_curriculum,
    foot_slip_curriculum,
    foot_slip_penalty,
    global_body_angular_velocity_tracking_exp,
    global_body_linear_velocity_tracking_exp,
    global_body_orientation_tracking_exp,
    global_body_position_tracking_exp,
    joint_position_tracking_exp,
    joint_velocity_tracking_exp,
    local_five_point_position_tracking_exp,
    local_foot_orientation_tracking_exp,
    survival_reward,
)
from .terminations import (
    body_orientation_tracking_failure,
    body_position_tracking_failure,
    motion_end,
    pelvis_position_tracking_failure,
    projected_gravity_tracking_failure,
    root_orientation_tracking_failure,
    root_position_tracking_failure,
)

__all__ = [name for name in globals() if not name.startswith("_")]

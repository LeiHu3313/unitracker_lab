"""MDP terms for Extreme-RGMT."""

from isaaclab.envs.mdp import (
    push_by_setting_velocity,
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    time_out,
    undesired_contacts,
)

from .actions import ReferenceResidualJointPositionActionCfg
from .commands import MotionCommand, MotionCommandCfg
from .events import randomize_motor_strength, randomize_motor_zero_offset
from .observations import critic_privileged_state, previous_action, proprioception, reference_window
from .rewards import (
    action_rate_penalty,
    body_angular_velocity_tracking_exp,
    body_linear_velocity_tracking_exp,
    body_orientation_tracking_exp,
    body_position_tracking_exp,
    controlled_joint_position_limit_penalty,
    foot_slip_penalty,
    global_body_orientation_tracking_exp,
)
from .terminations import (
    motion_end,
    pelvis_position_tracking_failure,
    projected_gravity_tracking_failure,
)

__all__ = [name for name in globals() if not name.startswith("_")]

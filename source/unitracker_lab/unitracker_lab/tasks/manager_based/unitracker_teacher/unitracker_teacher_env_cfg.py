"""G1-only UniTracker Stage-1 privileged teacher environment."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from unitracker_lab.assets.g1 import G1_29DOF_CFG, G1_ACTION_SCALE

from . import mdp
from .contracts import (
    ASSET_DR_RANGES,
    CONTROL_DECIMATION,
    FOOT_CONTACT_FORCE_THRESHOLD_N,
    G1_CONTROLLED_JOINT_NAMES,
    G1_FOOT_BODY_NAMES,
    G1_LOCAL_FIVE_POINT_BODY_NAMES,
    G1_REWARD_BODY_NAMES,
    G1_REWARD_JOINT_NAMES,
    PHYSICS_DT,
    PUSH_EVENT_SPECS,
    REGULARIZATION_REWARD_WEIGHTS,
    TERMINATION_SPECS,
    TRACKING_REWARD_SPECS,
)


@configclass
class TeacherSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        env_spacing=2.5,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class TeacherPlaySceneCfg(TeacherSceneCfg):
    """Teacher scene with physics-free reference markers for policy playback."""

    reference_robot: ArticulationCfg | None = None


@configclass
class TeacherCommandsCfg:
    motion = mdp.MotionCommandCfg(motion_file=MISSING)


@configclass
class TeacherActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G1_CONTROLLED_JOINT_NAMES),
        preserve_order=True,
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class TeacherObservationsCfg:
    @configclass
    class TeacherStateCfg(ObsGroup):
        state = ObsTerm(func=mdp.teacher_state_observation, params={"command_name": "motion"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherReferenceCfg(ObsGroup):
        reference = ObsTerm(func=mdp.teacher_reference_observation, params={"command_name": "motion"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    teacher_state: TeacherStateCfg = TeacherStateCfg()
    teacher_reference: TeacherReferenceCfg = TeacherReferenceCfg()


@configclass
class TeacherEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": ASSET_DR_RANGES["static_friction"],
            "dynamic_friction_range": ASSET_DR_RANGES["dynamic_friction"],
            "restitution_range": ASSET_DR_RANGES["restitution"],
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    torso_pelvis_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
            "com_range": {
                "x": ASSET_DR_RANGES["torso_pelvis_com_x"],
                "y": ASSET_DR_RANGES["torso_pelvis_com_yz"],
                "z": ASSET_DR_RANGES["torso_pelvis_com_yz"],
            },
        },
    )
    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": ASSET_DR_RANGES["link_mass_scale"],
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    external_push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_EVENT_SPECS["interval_range_s"],
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": PUSH_EVENT_SPECS["velocity_range"],
        },
    )


@configclass
class TeacherRewardsCfg:
    torso_position = RewTerm(
        func=mdp.global_body_position_tracking_exp,
        weight=TRACKING_REWARD_SPECS["torso_position"]["weight"],
        params={
            "command_name": "motion",
            "body_name": "torso_link",
            "sigma": TRACKING_REWARD_SPECS["torso_position"]["sigma"],
        },
    )
    torso_orientation = RewTerm(
        func=mdp.global_body_orientation_tracking_exp,
        weight=TRACKING_REWARD_SPECS["torso_orientation"]["weight"],
        params={
            "command_name": "motion",
            "body_name": "torso_link",
            "sigma": TRACKING_REWARD_SPECS["torso_orientation"]["sigma"],
        },
    )
    torso_linear_velocity = RewTerm(
        func=mdp.global_body_linear_velocity_tracking_exp,
        weight=TRACKING_REWARD_SPECS["torso_linear_velocity"]["weight"],
        params={
            "command_name": "motion",
            "body_name": "torso_link",
            "sigma": TRACKING_REWARD_SPECS["torso_linear_velocity"]["sigma"],
        },
    )
    torso_angular_velocity = RewTerm(
        func=mdp.global_body_angular_velocity_tracking_exp,
        weight=TRACKING_REWARD_SPECS["torso_angular_velocity"]["weight"],
        params={
            "command_name": "motion",
            "body_name": "torso_link",
            "sigma": TRACKING_REWARD_SPECS["torso_angular_velocity"]["sigma"],
        },
    )
    local_five_point_position = RewTerm(
        func=mdp.local_five_point_position_tracking_exp,
        weight=TRACKING_REWARD_SPECS["local_five_point_position"]["weight"],
        params={
            "command_name": "motion",
            "body_names": list(G1_LOCAL_FIVE_POINT_BODY_NAMES),
            "sigma": TRACKING_REWARD_SPECS["local_five_point_position"]["sigma"],
        },
    )
    local_foot_orientation = RewTerm(
        func=mdp.local_foot_orientation_tracking_exp,
        weight=TRACKING_REWARD_SPECS["local_foot_orientation"]["weight"],
        params={
            "command_name": "motion",
            "body_names": list(G1_FOOT_BODY_NAMES),
            "sigma": TRACKING_REWARD_SPECS["local_foot_orientation"]["sigma"],
        },
    )
    body_position = RewTerm(
        func=mdp.body_position_tracking_exp,
        weight=TRACKING_REWARD_SPECS["body_position"]["weight"],
        params={
            "command_name": "motion",
            "body_names": list(G1_REWARD_BODY_NAMES),
            "sigma": TRACKING_REWARD_SPECS["body_position"]["sigma"],
        },
    )
    body_orientation = RewTerm(
        func=mdp.body_orientation_tracking_exp,
        weight=TRACKING_REWARD_SPECS["body_orientation"]["weight"],
        params={
            "command_name": "motion",
            "body_names": list(G1_REWARD_BODY_NAMES),
            "sigma": TRACKING_REWARD_SPECS["body_orientation"]["sigma"],
        },
    )
    joint_position = RewTerm(
        func=mdp.joint_position_tracking_exp,
        weight=TRACKING_REWARD_SPECS["joint_position"]["weight"],
        params={
            "command_name": "motion",
            "joint_names": list(G1_REWARD_JOINT_NAMES),
            "sigma": TRACKING_REWARD_SPECS["joint_position"]["sigma"],
        },
    )
    joint_velocity = RewTerm(
        func=mdp.joint_velocity_tracking_exp,
        weight=TRACKING_REWARD_SPECS["joint_velocity"]["weight"],
        params={
            "command_name": "motion",
            "joint_names": list(G1_REWARD_JOINT_NAMES),
            "sigma": TRACKING_REWARD_SPECS["joint_velocity"]["sigma"],
        },
    )
    body_linear_velocity = RewTerm(
        func=mdp.body_linear_velocity_tracking_exp,
        weight=TRACKING_REWARD_SPECS["body_linear_velocity"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["body_linear_velocity"]["sigma"]},
    )
    body_angular_velocity = RewTerm(
        func=mdp.body_angular_velocity_tracking_exp,
        weight=TRACKING_REWARD_SPECS["body_angular_velocity"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["body_angular_velocity"]["sigma"]},
    )
    action_rate = RewTerm(
        func=mdp.action_rate_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["action_rate"],
    )
    controlled_joint_velocity = RewTerm(
        func=mdp.controlled_joint_velocity_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["controlled_joint_velocity"],
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(G1_CONTROLLED_JOINT_NAMES), preserve_order=True),
        },
    )
    controlled_joint_position_limits = RewTerm(
        func=mdp.controlled_joint_position_limit_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["controlled_joint_position_limits"],
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(G1_CONTROLLED_JOINT_NAMES), preserve_order=True),
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["foot_slip"],
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(G1_FOOT_BODY_NAMES), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(G1_FOOT_BODY_NAMES), preserve_order=True),
            "threshold": FOOT_CONTACT_FORCE_THRESHOLD_N,
        },
    )
    early_termination = RewTerm(
        func=mdp.early_termination_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["early_termination"],
    )
    survival = RewTerm(
        func=mdp.survival_reward,
        weight=REGULARIZATION_REWARD_WEIGHTS["survival"],
    )


@configclass
class TeacherTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_end = DoneTerm(func=mdp.motion_end, time_out=True, params={"command_name": "motion"})
    root_position = DoneTerm(
        func=mdp.root_position_tracking_failure,
        time_out=True,
        params={
            "command_name": "motion",
            "body_name": "torso_link",
            **TERMINATION_SPECS["root_position"],
        },
    )
    root_orientation = DoneTerm(
        func=mdp.root_orientation_tracking_failure,
        params={
            "command_name": "motion",
            "body_name": "torso_link",
            **TERMINATION_SPECS["root_orientation"],
        },
    )
    body_position = DoneTerm(
        func=mdp.body_position_tracking_failure,
        params={
            "command_name": "motion",
            "body_names": list(G1_REWARD_BODY_NAMES),
            "anchor_body_name": "torso_link",
            **TERMINATION_SPECS["body_position"],
        },
    )
    body_orientation = DoneTerm(
        func=mdp.body_orientation_tracking_failure,
        params={
            "command_name": "motion",
            "body_names": list(G1_REWARD_BODY_NAMES),
            "anchor_body_name": "torso_link",
            **TERMINATION_SPECS["body_orientation"],
        },
    )


@configclass
class UnitrackerTeacherEnvCfg(ManagerBasedRLEnvCfg):
    """Production G1 Stage-1 teacher configuration."""

    scene: TeacherSceneCfg = TeacherSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: TeacherActionsCfg = TeacherActionsCfg()
    commands: TeacherCommandsCfg = TeacherCommandsCfg()
    events: TeacherEventsCfg = TeacherEventsCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TeacherTerminationsCfg = TeacherTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = CONTROL_DECIMATION
        self.episode_length_s = 20.0
        self.sim.dt = PHYSICS_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (1.8, 1.8, 1.3)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class UnitrackerTeacherPlayEnvCfg(UnitrackerTeacherEnvCfg):
    """Deterministic clip replay with all Stage-1 randomization disabled."""

    scene: TeacherPlaySceneCfg = TeacherPlaySceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.commands.motion.sampling_mode = "eval"
        # Green target-body markers are rendered by ``MotionCommand``.  Unlike
        # an articulated ghost robot, these USD point instancers have no
        # colliders, masses, or contact response.
        self.commands.motion.debug_vis = True
        self.events.physics_material = None
        self.events.torso_pelvis_com = None
        self.events.link_mass = None
        self.events.external_push = None

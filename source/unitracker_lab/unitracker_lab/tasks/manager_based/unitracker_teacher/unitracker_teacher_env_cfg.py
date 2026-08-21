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
    G1_CONTROLLED_JOINT_NAMES,
    G1_FOOT_BODY_NAMES,
    G1_LOCAL_FIVE_POINT_BODY_NAMES,
    G1_ROOT_BODY_NAME,
    G1_TERMINATION_KEY_BODY_NAMES,
    PHYSICS_DT,
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
    class TeacherCfg(ObsGroup):
        oracle = ObsTerm(func=mdp.teacher_oracle_observation, params={"command_name": "motion"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        oracle = ObsTerm(func=mdp.teacher_oracle_observation, params={"command_name": "motion"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()
    critic: CriticCfg = CriticCfg()


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


@configclass
class TeacherRewardsCfg:
    base_position = RewTerm(
        func=mdp.base_position_tracking_exp,
        weight=TRACKING_REWARD_SPECS["base_position"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["base_position"]["sigma"]},
    )
    base_orientation = RewTerm(
        func=mdp.base_orientation_tracking_exp,
        weight=TRACKING_REWARD_SPECS["base_orientation"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["base_orientation"]["sigma"]},
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
    body_position = RewTerm(
        func=mdp.body_position_tracking_exp,
        weight=TRACKING_REWARD_SPECS["body_position"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["body_position"]["sigma"]},
    )
    body_orientation = RewTerm(
        func=mdp.body_orientation_tracking_exp,
        weight=TRACKING_REWARD_SPECS["body_orientation"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["body_orientation"]["sigma"]},
    )
    joint_position = RewTerm(
        func=mdp.joint_position_tracking_exp,
        weight=TRACKING_REWARD_SPECS["joint_position"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["joint_position"]["sigma"]},
    )
    joint_velocity = RewTerm(
        func=mdp.joint_velocity_tracking_exp,
        weight=TRACKING_REWARD_SPECS["joint_velocity"]["weight"],
        params={"command_name": "motion", "sigma": TRACKING_REWARD_SPECS["joint_velocity"]["sigma"]},
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
    torso_orientation = RewTerm(
        func=mdp.torso_orientation_tracking_exp,
        weight=TRACKING_REWARD_SPECS["torso_orientation"]["weight"],
        params={
            "command_name": "motion",
            "body_name": "torso_link",
            "sigma": TRACKING_REWARD_SPECS["torso_orientation"]["sigma"],
        },
    )
    action_rate = RewTerm(
        func=mdp.action_rate_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["action_rate"],
    )
    controlled_joint_torque = RewTerm(
        func=mdp.controlled_joint_torque_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["controlled_joint_torque"],
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
            "threshold": 1.0,
        },
    )
    early_termination = RewTerm(
        func=mdp.early_termination_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["early_termination"],
    )


@configclass
class TeacherTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_end = DoneTerm(func=mdp.motion_end, time_out=True, params={"command_name": "motion"})
    projected_gravity = DoneTerm(
        func=mdp.projected_gravity_tracking_failure,
        params={"command_name": "motion", **TERMINATION_SPECS["projected_gravity"]},
    )
    key_body_height = DoneTerm(
        func=mdp.key_body_height_tracking_failure,
        params={
            "command_name": "motion",
            "body_names": list(G1_TERMINATION_KEY_BODY_NAMES),
            **TERMINATION_SPECS["key_body_height"],
        },
    )
    pelvis_position = DoneTerm(
        func=mdp.pelvis_position_tracking_failure,
        params={
            "command_name": "motion",
            "body_name": G1_ROOT_BODY_NAME,
            **TERMINATION_SPECS["pelvis_position"],
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
        self.episode_length_s = 10.0
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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.commands.motion.sampling_mode = "eval"
        self.commands.motion.debug_vis = True
        self.events.physics_material = None
        self.events.torso_pelvis_com = None
        self.events.link_mass = None

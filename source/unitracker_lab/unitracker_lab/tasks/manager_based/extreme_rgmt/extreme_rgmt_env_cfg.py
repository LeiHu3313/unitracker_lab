"""Standalone G1 environment for the Extreme-RGMT reproduction."""

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

from unitracker_lab.assets.g1 import G1_29DOF_CFG

from . import mdp
from .contracts import (
    ASSET_DR_RANGES,
    CONTROL_DECIMATION,
    FOOT_CONTACT_FORCE_THRESHOLD_N,
    G1_CONTROLLED_JOINT_NAMES,
    G1_FOOT_BODY_NAMES,
    G1_ROOT_BODY_NAME,
    HISTORY_LENGTH,
    PHYSICS_DT,
    REGULARIZATION_REWARD_WEIGHTS,
    TERMINATION_SPECS,
    TRACKING_REWARD_SPECS,
)


@configclass
class ExtremeRGMTSceneCfg(InteractiveSceneCfg):
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
class ExtremeRGMTCommandsCfg:
    motion = mdp.MotionCommandCfg(motion_file=MISSING)


@configclass
class ExtremeRGMTActionsCfg:
    joint_pos = mdp.ReferenceResidualJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(G1_CONTROLLED_JOINT_NAMES),
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
        command_name="motion",
    )


@configclass
class ExtremeRGMTObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprioception = ObsTerm(
            func=mdp.proprioception,
            params={"command_name": "motion", "enable_noise": True},
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        previous_action = ObsTerm(
            func=mdp.previous_action,
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        reference_window = ObsTerm(
            func=mdp.reference_window,
            params={"command_name": "motion", "enable_noise": True},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        proprioception = ObsTerm(
            func=mdp.proprioception,
            params={"command_name": "motion", "enable_noise": True},
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        previous_action = ObsTerm(
            func=mdp.previous_action,
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        reference_window = ObsTerm(
            func=mdp.reference_window,
            params={"command_name": "motion", "enable_noise": True},
        )
        privileged_state = ObsTerm(func=mdp.critic_privileged_state, params={"command_name": "motion"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ExtremeRGMTEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": ASSET_DR_RANGES["ground_friction"],
            "dynamic_friction_range": ASSET_DR_RANGES["ground_friction"],
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    added_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[G1_ROOT_BODY_NAME]),
            "mass_distribution_params": ASSET_DR_RANGES["added_base_mass_kg"],
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[G1_ROOT_BODY_NAME]),
            "com_range": {
                "x": ASSET_DR_RANGES["base_com_x_m"],
                "y": ASSET_DR_RANGES["base_com_yz_m"],
                "z": ASSET_DR_RANGES["base_com_yz_m"],
            },
        },
    )
    motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(G1_CONTROLLED_JOINT_NAMES)),
            "scale_range": ASSET_DR_RANGES["motor_strength_scale"],
        },
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(G1_CONTROLLED_JOINT_NAMES)),
            "stiffness_distribution_params": ASSET_DR_RANGES["pd_gain_scale"],
            "damping_distribution_params": ASSET_DR_RANGES["pd_gain_scale"],
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    motor_zero_offset = EventTerm(
        func=mdp.randomize_motor_zero_offset,
        mode="startup",
        params={"offset_range": ASSET_DR_RANGES["motor_zero_offset_rad"]},
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(G1_CONTROLLED_JOINT_NAMES)),
            "armature_distribution_params": ASSET_DR_RANGES["joint_armature_scale"],
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # The paper publishes the interval but not the impulse magnitude. The
    # velocity range below is an explicit, serialized reproduction choice.
    external_push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=ASSET_DR_RANGES["external_push_interval_s"],
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        },
    )


@configclass
class ExtremeRGMTRewardsCfg:
    anchor_orientation = RewTerm(
        func=mdp.global_body_orientation_tracking_exp,
        weight=TRACKING_REWARD_SPECS["anchor_orientation"]["weight"],
        params={
            "command_name": "motion",
            "body_name": G1_ROOT_BODY_NAME,
            "sigma": TRACKING_REWARD_SPECS["anchor_orientation"]["sigma"],
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
    joint_position_limits = RewTerm(
        func=mdp.controlled_joint_position_limit_penalty,
        weight=REGULARIZATION_REWARD_WEIGHTS["joint_position_limits"],
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(G1_CONTROLLED_JOINT_NAMES), preserve_order=True),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=REGULARIZATION_REWARD_WEIGHTS["undesired_contacts"],
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_rubber_hand$)(?!right_rubber_hand$).+$"
                ],
            ),
            "threshold": FOOT_CONTACT_FORCE_THRESHOLD_N,
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


@configclass
class ExtremeRGMTTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_end = DoneTerm(func=mdp.motion_end, time_out=True, params={"command_name": "motion"})
    projected_gravity = DoneTerm(
        func=mdp.projected_gravity_tracking_failure,
        params={"command_name": "motion", **TERMINATION_SPECS["projected_gravity"]},
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
class ExtremeRGMTEnvCfg(ManagerBasedRLEnvCfg):
    """Role-aware Extreme-RGMT environment configured by the launcher."""

    scene: ExtremeRGMTSceneCfg = ExtremeRGMTSceneCfg(num_envs=8192, env_spacing=2.5)
    observations: ExtremeRGMTObservationsCfg = ExtremeRGMTObservationsCfg()
    actions: ExtremeRGMTActionsCfg = ExtremeRGMTActionsCfg()
    commands: ExtremeRGMTCommandsCfg = ExtremeRGMTCommandsCfg()
    events: ExtremeRGMTEventsCfg = ExtremeRGMTEventsCfg()
    rewards: ExtremeRGMTRewardsCfg = ExtremeRGMTRewardsCfg()
    terminations: ExtremeRGMTTerminationsCfg = ExtremeRGMTTerminationsCfg()

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
        self.commands.motion.acquisition_fraction = 0.8

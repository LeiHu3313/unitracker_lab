"""Play a G1 Stage-1 teacher checkpoint on deterministic reference clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--motion", required=True, type=Path, help="Prepared G1 NPZ, recursive directory, or .txt/.lst manifest."
)
parser.add_argument(
    "--checkpoint_path", type=Path, default=None, help="Exact model .pt; otherwise resolve load_run/checkpoint."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="Exit after N steps; 0 runs while the app is open.")
parser.add_argument("--task", default="Unitracker_Teacher-Play-v0", choices=["Unitracker_Teacher-Play-v0"])
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument(
    "--reference_robot",
    action="store_true",
    help="Show the tracked reference pose as a translucent, non-colliding G1 robot.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import unitracker_lab.tasks  # noqa: F401, E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitracker_lab.assets.g1 import G1_29DOF_CFG  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.motion_schema import (  # noqa: E402
    load_and_validate_motion_dataset,
)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _make_reference_robot_cfg():
    """Create a visual-only G1 articulation for reference-pose playback."""

    spawn_cfg = G1_29DOF_CFG.spawn.replace(
        activate_contact_sensors=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=G1_29DOF_CFG.spawn.rigid_props.replace(disable_gravity=True),
        articulation_props=G1_29DOF_CFG.spawn.articulation_props.replace(enabled_self_collisions=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.85, 0.25),
            opacity=0.35,
        ),
    )
    return G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/ReferenceRobot",
        spawn=spawn_cfg,
    )


class _ReferenceRobotVisualizer:
    """Keep the visual-only articulation at MotionCommand's target frame."""

    def __init__(self, env) -> None:
        self._env = env.unwrapped
        self._robot: Articulation = self._env.scene["reference_robot"]
        self._command = self._env.command_manager.get_term("motion")
        controlled_ids, controlled_names = self._robot.find_joints(
            self._command.cfg.controlled_joint_names, preserve_order=True
        )
        locked_ids, locked_names = self._robot.find_joints(
            self._command.cfg.locked_joint_names, preserve_order=True
        )
        if tuple(controlled_names) != tuple(self._command.cfg.controlled_joint_names):
            raise ValueError(f"Reference robot controlled joints do not match the motion contract: {controlled_names}")
        if tuple(locked_names) != tuple(self._command.cfg.locked_joint_names):
            raise ValueError(f"Reference robot locked joints do not match the motion contract: {locked_names}")
        self._controlled_joint_ids = controlled_ids
        self._locked_joint_ids = locked_ids
        self._root_body_index = self._command.cfg.body_names.index(self._command.cfg.root_body_name)
        self._joint_pos = self._robot.data.default_joint_pos.clone()
        self._joint_vel = torch.zeros_like(self._joint_pos)
        self._locked_joint_positions = torch.as_tensor(
            self._command.cfg.locked_joint_positions,
            dtype=self._joint_pos.dtype,
            device=self._joint_pos.device,
        )
        self._root_velocity = torch.zeros(
            (self._env.num_envs, 6), dtype=self._joint_pos.dtype, device=self._joint_pos.device
        )

    def update(self) -> None:
        """Write a stationary target pose before the environment renders this control step."""

        self._joint_pos.copy_(self._robot.data.default_joint_pos)
        self._joint_pos[:, self._controlled_joint_ids] = self._command.target_ref_joint_pos
        self._joint_pos[:, self._locked_joint_ids] = self._locked_joint_positions
        root_state = torch.cat(
            (
                self._command.target_ref_body_pos_w[:, self._root_body_index],
                self._command.target_ref_body_quat_w[:, self._root_body_index],
                self._root_velocity,
            ),
            dim=-1,
        )
        self._robot.write_root_state_to_sim(root_state)
        self._robot.write_joint_state_to_sim(self._joint_pos, self._joint_vel)
        self._robot.set_joint_position_target(self._joint_pos)
        self._robot.set_joint_velocity_target(self._joint_vel)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    motion = load_and_validate_motion_dataset(args_cli.motion)
    motion_path = motion.path
    motion_count = motion.num_motions
    motion_frame_count = motion.frame_count
    motion_fps = motion.fps
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = str(motion_path)
    # The full reference robot replaces the point markers to keep comparisons readable.
    env_cfg.commands.motion.debug_vis = not args_cli.headless and not args_cli.reference_robot
    if args_cli.reference_robot:
        env_cfg.scene.reference_robot = _make_reference_robot_cfg()
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    if args_cli.checkpoint_path is not None:
        checkpoint = args_cli.checkpoint_path.expanduser().resolve()
    else:
        log_root = _REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
        checkpoint = Path(get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint))
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    print(f"[INFO] Motion input: {motion_path}")
    print(f"[INFO] Motion clips/frames/FPS: {motion_count}/{motion_frame_count}/{motion_fps:g}")
    print(f"[INFO] Checkpoint: {checkpoint}")

    # Avoid retaining the preflight arrays while MotionCommand loads the data.
    del motion

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(checkpoint.parent / "videos" / "play"),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        reference_visualizer = _ReferenceRobotVisualizer(env) if args_cli.reference_robot else None
        observations = env.get_observations()
        step = 0
        with torch.inference_mode():
            while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
                if reference_visualizer is not None:
                    reference_visualizer.update()
                actions = policy(observations)
                observations, _, _, _ = env.step(actions)
                step += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

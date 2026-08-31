"""Visually replay one prepared 29-DoF G1 motion without physics or a policy.

Exactly one green G1 is spawned. Every rendered frame writes the reference root
and all 29 joint states directly to the simulator; this script never advances
physics, PD targets, a task environment, rewards, or a policy checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", required=True, type=Path, help="One prepared 29-DoF G1 NPZ clip.")
parser.add_argument("--steps", type=int, default=0, help="Frames to replay; 0 plays the entire clip once.")
parser.add_argument(
    "--loop", action="store_true", help="Repeat the clip until the viewer is closed or --steps is reached."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from unitracker_lab.assets.g1 import G1_29DOF_CFG  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.contracts import G1_ALL_JOINT_NAMES  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.motion_schema import (  # noqa: E402
    load_and_validate_motion_dataset,
)
from unitracker_lab.tasks.manager_based.unitracker_teacher.unitracker_teacher_env_cfg import (  # noqa: E402
    TeacherSceneCfg,
)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def _reference_robot_cfg():
    """Return the sole, visual-only robot configuration for this viewer."""

    spawn_cfg = G1_29DOF_CFG.spawn.replace(
        activate_contact_sensors=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=G1_29DOF_CFG.spawn.rigid_props.replace(disable_gravity=True),
        articulation_props=G1_29DOF_CFG.spawn.articulation_props.replace(enabled_self_collisions=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.25), opacity=1.0),
    )
    return G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/ReferenceRobot", spawn=spawn_cfg)


def main() -> None:
    motion = load_and_validate_motion_dataset(args_cli.motion)
    if motion.num_motions != 1:
        raise ValueError(f"Reference playback accepts exactly one NPZ clip; got {motion.num_motions}: {motion.path}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / motion.fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = TeacherSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = _reference_robot_cfg()
    scene_cfg.contact_forces = None
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    joint_ids, joint_names = robot.find_joints(list(G1_ALL_JOINT_NAMES), preserve_order=True)
    if tuple(joint_names) != G1_ALL_JOINT_NAMES:
        raise ValueError(f"Live G1 joints do not match the 29-DoF motion contract: {joint_names}")
    joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=sim.device)

    root_pos = torch.as_tensor(motion.body_pos_w[:, 0], dtype=torch.float32, device=sim.device)
    root_quat = torch.as_tensor(motion.body_quat_w[:, 0], dtype=torch.float32, device=sim.device)
    root_lin_vel = torch.as_tensor(motion.body_lin_vel_w[:, 0], dtype=torch.float32, device=sim.device)
    root_ang_vel = torch.as_tensor(motion.body_ang_vel_w[:, 0], dtype=torch.float32, device=sim.device)
    joint_pos_ref = torch.as_tensor(motion.joint_pos, dtype=torch.float32, device=sim.device)
    joint_vel_ref = torch.as_tensor(motion.joint_vel, dtype=torch.float32, device=sim.device)

    total_steps = motion.frame_count if args_cli.steps == 0 else args_cli.steps
    if total_steps <= 0:
        raise ValueError("--steps must be positive, or the motion must contain at least one frame")
    if not args_cli.loop:
        total_steps = min(total_steps, motion.frame_count)
    print(f"[INFO] Pure reference replay: {motion.path} ({motion.frame_count} frames at {motion.fps:g} Hz)")

    step = 0
    while simulation_app.is_running() and step < total_steps:
        frame = step % motion.frame_count
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos[frame] + scene.env_origins[:, :3]
        root_state[:, 3:7] = root_quat[frame]
        root_state[:, 7:10] = root_lin_vel[frame]
        root_state[:, 10:13] = root_ang_vel[frame]
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_ids] = joint_pos_ref[frame]
        joint_vel[:, joint_ids] = joint_vel_ref[frame]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())
        look_at = root_state[0, :3].detach().cpu().numpy()
        sim.set_camera_view(eye=look_at + (2.0, 2.0, 0.8), target=look_at)
        step += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

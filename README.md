# UniTracker Lab

[中文说明](README_CN.md)

Isaac Lab project for  UniTracker environments and learning algorithms.

## Environment

- Linux
- NVIDIA GPU and a driver compatible with Isaac Sim
- Python 3.11 recommended (Python 3.10 or newer is supported)
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- RSL-RL v3.1.2 (included as a Git submodule)

## Setup

Clone the project with its RSL-RL submodule:

```bash
git clone --recurse-submodules https://github.com/LeiHu3313/unitracker_lab.git
cd unitracker_lab
```

If the project was cloned without submodules:

```bash
git submodule update --init --recursive
```

Activate the Python environment used by Isaac Lab, then install the project and its local RSL-RL package:

```bash
python -m pip install -e source/unitracker_lab
python -m pip install -e rsl_rl
```

## G1 Teacher

Train on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/rsl_rl/train.py \
  --task Unitracker_Teacher-v0 \
  --motion data/g1_lafan_40_prepared_29dof \
  --num_envs 1024 \
  --max_iterations 50000 \
  --seed 42 \
  --distributed \
  --headless
```

Play a checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task Unitracker_Teacher-Play-v0 \
  --motion data/g1_lafan_40_prepared_29dof \
  --checkpoint /absolute/path/to/model_2000.pt \
  --num_envs 1
```

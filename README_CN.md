# UniTracker Lab

[English](README.md)

用于实现 UniTracker 环境与学习算法的 Isaac Lab 项目。

## 环境要求

- Linux
- NVIDIA GPU，以及与 Isaac Sim 兼容的驱动
- 推荐 Python 3.11（支持 Python 3.10 及以上版本）
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- RSL-RL v3.1.2（以 Git submodule 引入）

## 环境配置

克隆项目及 RSL-RL submodule：

```bash
git clone --recurse-submodules https://github.com/LeiHu3313/unitracker_lab.git
cd unitracker_lab
```

如果克隆时没有拉取 submodule：

```bash
git submodule update --init --recursive
```

激活 Isaac Lab 使用的 Python 环境，然后安装本项目和本地 RSL-RL：

```bash
python -m pip install -e source/unitracker_lab
python -m pip install -e rsl_rl
```

## G1 Teacher 训练与播放

以下命令假定已激活可直接运行 Isaac Lab 的 Python 环境，并在仓库根目录执行。当前
Teacher 使用完整 29 DoF G1 reference；`--motion` 必须指向 prepared NPZ、包含 NPZ
的目录，或 `.txt/.lst` manifest。

先以小规模单卡训练验证环境、数据和配置：

```bash
cd /home/hul/workspace/hl/my_projects/unitracker_lab

python scripts/rsl_rl/train.py \
  --task Unitracker_Teacher-v0 \
  --motion data/g1_lafan_40 \
  --num_envs 64 \
  --max_iterations 10 \
  --seed 42 \
  --headless
```

4 卡正式训练。`--num_envs` 为每张卡的环境数；下例总计 4096 个环境：

```bash
cd /home/hul/workspace/hl/my_projects/unitracker_lab

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  scripts/rsl_rl/train.py \
  --task Unitracker_Teacher-v0 \
  --motion data/g1_lafan_40 \
  --num_envs 1024 \
  --max_iterations 50000 \
  --seed 42 \
  --distributed \
  --headless
```

训练输出在 `logs/rsl_rl/unitracker_teacher/<时间>_g1_stage1/`。播放 checkpoint 时使用
`Unitracker_Teacher-Play-v0`，并传入训练所用的同一份 motion 数据：

```bash
cd /home/hul/workspace/hl/my_projects/unitracker_lab

python scripts/rsl_rl/play.py \
  --task Unitracker_Teacher-Play-v0 \
  --motion data/g1_lafan_40 \
  --checkpoint /absolute/path/to/model_2000.pt \
  --num_envs 1
```

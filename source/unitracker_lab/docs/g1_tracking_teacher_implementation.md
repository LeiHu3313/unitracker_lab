# G1 Teacher 当前训练配置

本文描述当前 `Unitracker_Teacher-v0` 的实际训练合同。训练时以当前代码和每次 run 保存的 `params/env.yaml`、`params/agent.yaml` 为准。

## 1. 运行链路

```text
data/g1_lafan_40/*.npz (50 Hz, 29 joint, 17 body)
  -> MotionCommand (clip RSI, k -> k+1 reference)
  -> ManagerBasedRLEnv: Unitracker_Teacher-v0
  -> teacher_state (677D) + teacher_reference (748D)
  -> PPO actor / critic (each 1425D, fully privileged)
  -> 29D joint-position action
  -> G1 implicit PD
  -> tracking reward, termination, reset / adaptive resampling
```

Gym task:

| ID | 用途 |
|---|---|
| `Unitracker_Teacher-v0` | 训练；平面、随机化、adaptive motion sampling。 |
| `Unitracker_Teacher-Play-v0` | checkpoint 播放；单环境、绿色 reference G1、关闭随机化和 push。 |

入口为 `scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py` 和 `scripts/rsl_rl/play_motion_reference.py`。`--motion` 可传一个 NPZ、NPZ 目录或 `.txt/.lst` manifest。

## 2. 机器人、时序与动作

- 机器人：Unitree G1 29DoF，全部 29 个物理关节都由策略 action/reference 控制；六个 wrist 不再锁定。
- physics：`dt = 0.005 s`（200 Hz）。
- control：decimation `4`，因此策略和数据均为 `0.02 s` / `50 Hz`。
- action：29D normalized joint-position residual，经每关节 `G1_ACTION_SCALE` 和 default joint offset 写入 implicit PD target。
- reference 时序：状态 phase 为 `k`，奖励和下一个目标为 `k + 1`；每条 clip 最后一帧不作为 RSI 起点，reference 不跨 clip。

关节、body、action、observation 的稳定顺序都在 `contracts.py` 中定义。修改其中 ordered contract 会改变 checkpoint ABI，不能加载旧模型。

## 3. 训练数据

当前训练集：`data/g1_lafan_40`。

| 项目 | 当前值 |
|---|---|
| clips / frames | 40 / 441140 |
| fps | 50 |
| joint state | `joint_pos`, `joint_vel`，各 `[T, 29]` |
| body state | position / WXYZ quaternion / linear velocity / angular velocity，各为 17 body |
| 存储 | Git LFS |

完整字段、body 顺序、生成来源见 `data/g1_lafan_40/dataset.yaml`。加载器会严格验证 50 Hz、29 joint 名称、17 body 名称、shape、有限值和 quaternion 归一化，再按合同顺序重排。缺少任一必需字段即拒绝启动。

17 个数据 body 为 pelvis、左右 hip-yaw / knee / ankle-roll、waist-roll、torso、左右 shoulder-yaw / elbow / wrist-yaw / rubber-hand。数据保留 waist 与 rubber hand，便于后续 reward 或操作任务使用。

## 4. Motion RSI 与自适应采样

每个环境复位时选择一个 clip 和合法 phase，直接将该帧 root/joint state 写入仿真，再追踪 `k -> k+1`。motion 结束为 timeout，不跨到下一个 clip。

训练默认 `sampling_mode="adaptive"`：

- 每条 motion 先分为 `0.25 s`（12 frame）bins；每条 clip 保持相同总采样质量，避免长 clip 主导。
- 采样分布为 50% phase-uniform + 50% difficulty-focused。
- difficulty 是 bin 的 failure rate 与连续 tracking error 的 EMA；误差使用 reward 的 sigma 归一化，并只用与 whole-body pose reward 一致的 14 body。
- 实际 tracking failure 时，80% 概率在同一 clip 回退随机 25--124 frame（0.5--2.48 s）后重新开始；距离 clip 末尾不足 50 frame 或 timeout 时改用全局 adaptive 重采样。
- 多卡时每个 PPO rollout 后 all-reduce 采样统计，再更新所有 rank 的同一分布。

## 5. Teacher 观测

Teacher 是全特权训练策略。actor 和 critic 使用完全相同的原始输入组：

```python
{"policy": ["teacher_state", "teacher_reference"],
 "critic": ["teacher_state", "teacher_reference"]}
```

RSL-RL 的 actor / critic normalizer 仍各自独立。两个 observation group 均不注入噪声，总维度为 `677 + 748 = 1425`。

### 5.1 `teacher_state`（677D）

| 项 | 维度 | 表达 |
|---|---:|---|
| root angular velocity history | 21 | 7 帧 × 3，当前 pelvis 完整坐标系。 |
| projected gravity history | 21 | 7 帧 × 3，pelvis 完整坐标系。 |
| joint position history | 203 | 7 帧 × 29，相对 default joint position。 |
| joint velocity history | 203 | 7 帧 × 29。 |
| previous action history | 87 | 3 帧 × 29。 |
| current body position | 42 | 14 body × 3；原点为当前 pelvis 的地面投影，轴为当前 pelvis 完整朝向。 |
| current body linear velocity | 42 | 14 body × 3，当前 pelvis 完整坐标系。 |
| applied action | 29 | action manager 当前 policy-unit action。 |
| applied joint torque | 29 | 全部 29 关节的仿真 applied torque。 |

state history offsets 为 `[0, 1, 2, 3, 4, 8, 16]`，action history offsets 为 `[0, 1, 2]`；复位时由 reference state 初始化，前序 action 置零。

### 5.2 `teacher_reference`（748D）

| 项 | 维度 | 表达 |
|---|---:|---|
| reference root position trajectory | 24 | 8 reference 时刻 × 3；reference 当前 pelvis 地面投影 + yaw-only 坐标系。 |
| reference root orientation trajectory | 48 | 8 × Rot6D；相对当前 robot pelvis 完整朝向。 |
| reference joint-position trajectory | 232 | 8 × 29。 |
| reference root position trajectory | 24 | 8 × 3；相对当前 robot pelvis 完整坐标系。 |
| body position error | 84 | 2 时刻 × 14 body × 3；robot/reference 分别在自身 pelvis-ground yaw frame 后相减。 |
| body orientation error | 168 | 2 × 14 × Rot6D；同一 yaw-only 语义。 |
| body linear velocity error | 84 | 2 × 14 × 3，世界系直接相减。 |
| body angular velocity error | 84 | 2 × 14 × 3，世界系直接相减。 |

reference offsets 为 `[-8, -4, -2, 0, 1, 2, 3, 4]`，feedback offsets 为 `[0, 1]`。所有 reference index 都在当前 clip 内 clamp。

14 个 observation body 为 pelvis、左右 hip-yaw / knee / ankle-roll、torso、左右 shoulder-yaw / elbow / wrist-yaw；不含 waist-roll 和左右 rubber-hand。

## 6. Reward

每个 tracking term 均为 exponential reward，使用配置中的 weight 和 sigma。`local` 指 reference 与 robot 各自在自己的 pelvis 完整 root frame（position）或 root-relative orientation（orientation）中比较；`global` 在世界系比较。

| 项 | weight | sigma | 范围 / 语义 |
|---|---:|---:|---|
| torso position / orientation | 0.5 / 0.5 | 0.30 / 0.40 | torso 世界位置 / 朝向。 |
| torso linear / angular velocity | 0.5 / 0.5 | 1.00 / 2.50 | torso 世界速度。 |
| local five-point position | 0.5 | 0.12 | torso、左右 ankle-roll、左右 rubber-hand。 |
| local foot orientation | 0.1 | 0.30 | 左右 ankle-roll 的 root-relative 完整朝向。 |
| body position / orientation | 1.0 / 1.0 | 0.30 / 0.40 | 14 body root-local pose。 |
| joint position / velocity | 0.5 / 0.5 | 0.25 / 2.50 | 25 joint：排除四个 ankle pitch/roll joint。 |
| body linear / angular velocity | 0.5 / 0.5 | 1.00 / 2.50 | 全部 17 motion body 的世界速度。 |

踝仍是 29D action/reference 的一部分；去掉的是踝 q/dq 的直接 reward，踝由 ankle-roll endpoint pose、foot orientation、body velocity、foot slip 和接触动力学共同约束。

正则项：action rate `-0.01`、controlled joint velocity `-1e-4`、joint limit `-10.0`、foot slip `-1.5`、early termination `-50.0`。没有 reward curriculum。

## 7. 终止与随机化

训练 episode 上限为 20 s，以下终止均会触发 reset：

| 终止 | 类型 | 条件 |
|---|---|---|
| `time_out` | timeout | 20 s。 |
| `motion_end` | timeout | 当前 clip 到末尾。 |
| `projected_gravity` | failure | robot 与 target pelvis projected gravity 差的 norm > `0.8`。 |
| `pelvis_position` | failure | pelvis 高度相对 target 的绝对误差 > `0.4 m`。 |

训练随机化：startup friction（static/dynamic `0.8--1.2`）、restitution（`0--0.15`）、pelvis/torso COM（各轴最多 `±0.01 m`）、全 link mass scale（`0.95--1.05`）。另有每 `4--8 s` 一次、xy 分量 `±0.15 m/s` 的 root velocity push。Play task 关闭上述全部随机化和 push。

## 8. PPO 与多卡

PPO 使用 24 rollout steps / env、50000 iteration、每 500 iteration 保存；actor 与 critic hidden dims 均为 `[1024, 512, 512, 256]`，ELU，learning rate `1e-3` adaptive schedule，`gamma=0.99`，`lambda=0.95`，5 epochs，4 minibatches，entropy coefficient `0.005`。

多卡启动使用 `torch.distributed.run` 加 `--distributed`。`--num_envs` 是**每个 rank**的环境数；两卡需要总计 3076 env 时，每卡应设为 1538。

`rsl_rl` 的 `OnPolicyRunner` 会先执行 `torch.cuda.set_device(LOCAL_RANK)`、再初始化 NCCL process group；首轮模型状态同步使用逐个 CUDA state tensor 的原地 broadcast，避免通过 `broadcast_object_list` 序列化 GPU `state_dict`。这两项作为多卡训练的健壮性和性能保障保留。

此外，当前 DSW 容器的 NCCL shared-memory transport 会在首次 collective 报 `invalid argument`；即使保留上述实现，该服务器启动多卡训练时仍需设置 `NCCL_SHM_DISABLE=1`，禁用 SHM 并改用网络 transport。这是服务器运行环境限制，不是 Teacher 或 RSL 算法逻辑差异。

两卡示例：

```bash
cd /mnt/workspace/Project_hul/unitracker_lab
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_SHM_DISABLE=1

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task Unitracker_Teacher-v0 \
  --motion data/g1_lafan_40 \
  --num_envs 1538 \
  --max_iterations 50000 \
  --seed 42 \
  --distributed \
  --headless
```

单卡 smoke：

```bash
python scripts/rsl_rl/train.py \
  --task Unitracker_Teacher-v0 \
  --motion data/g1_lafan_40 \
  --num_envs 64 \
  --max_iterations 5 \
  --seed 42 \
  --headless
```

## 9. 数据和策略播放

只查看一个 reference NPZ（绿色 G1，直接写 state，不运行 policy/physics）：

```bash
python scripts/rsl_rl/play_motion_reference.py \
  --motion data/g1_lafan_40/walk1_subject1.npz
```

播放 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task Unitracker_Teacher-Play-v0 \
  --motion data/g1_lafan_40 \
  --checkpoint /absolute/path/to/model_XXXX.pt
```

旧 789D / 855D observation checkpoint 与当前 1425D teacher 不兼容，不能 resume 或直接 play。

## 10. 关键实现位置

| 内容 | 位置 |
|---|---|
| task 注册、scene、action、observation group、reward、termination、DR | `unitracker_teacher_env_cfg.py` |
| joint/body/timing/observation/reward 合同 | `contracts.py` |
| motion loader 与 schema 校验 | `motion_schema.py` |
| RSI、reference、history、adaptive sampling | `mdp/commands.py`、`mdp/adaptive_sampling.py` |
| observation 的参考系和拼接 | `mdp/observations.py` |
| reward 实现 | `mdp/rewards.py` |
| PPO 配置 | `agents/rsl_rl_ppo_cfg.py` |
| 多卡 NCCL 初始化 | `rsl_rl/rsl_rl/runners/on_policy_runner.py` |
| 数据集元数据 | `data/g1_lafan_40/dataset.yaml` |

# Extreme-RGMT 方法调研与复现计划

## 1. 结论与复现边界

目标论文：Yubiao Ma 等，*Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control*，arXiv:2607.20110v1，2026-07-22。

截至 2026-08-24，项目页未公开训练代码、模型、数据、动作重定向流水线或完整配置。本分支采用 clean-room 复现，并保持为独立功能：

- 任务只注册 `Extreme-RGMT-Base-v0` 和 `Extreme-RGMT-Expansion-v0`；
- 单一入口 `scripts/rsl_rl/extreme_rgmt/train.py` 分别启动 Stage I 和 Stage II；
- 不依赖或保留 UniTracker Teacher 的任务、配置和训练脚本；
- 已实现 Stage II 的 PACE/STAR、角色感知采样、冻结参考策略和运行契约；
- 当前 Stage I 仍是可运行的 MLP 代理基座，尚未实现论文的 history/reference encoder 与 FSQ，不能据此声明端到端数值复现。

## 2. 论文方法

### 2.1 控制与 Stage I

控制频率为 50 Hz，底层 PD 为 500 Hz。论文 actor 使用：

- 10 帧 proprioception history：投影重力、base 角速度、关节位置与速度；
- 10 帧 previous-action history；
- 以当前时刻为中心的 21-token reference window；每个 token 含 reference base 线/角速度、重力方向和关节位姿；
- critic 额外使用 base 高度、body link 位姿、base 线速度等 privileged state。

策略输出 29 维 reference-residual joint position：

```text
q_target(t) = q_ref(t) + a(t)
tau(t) = Kp * (q_target(t) - q(t)) - Kd * qdot(t)
```

Stage I command encoder 将 state/action history 分别编码并交错送入 causal encoder，以 history embedding 查询 reference-window cross-attention；聚合结果拆成两个 32 维 token，经 FSQ 后与当前状态、上一动作一同送入 actor。

论文公开超参数：

| 项目 | 取值 |
| --- | --- |
| state/action/reference encoder | `[64,128,64]` / `[29,64,64]` / `[38,128,64]` |
| actor / critic | `[1024,1024,512,256]` / `[1024,1024,512,512]` |
| rollout / epochs / mini-batches | `24 / 5 / 4` |
| learning rate / target KL | `1e-3 adaptive / 0.01` |
| gamma / GAE lambda / PPO clip | `0.99 / 0.95 / 0.2` |
| entropy coefficient | `0.005` |

训练动作总计 3.096 h，来自 LAFAN1、AMASS 和 in-house Xsens，重定向到 G1 并重采样为 50 Hz。

### 2.2 Motion stratification

Stage I 完成后，将超过 10 s 的 motion 切为 10 s clip；每个 clip 随机 rollout 5 次。completion rate `>= 0.8` 进入 mastered set `D_m`，其余进入 challenging set `D_c`。论文报告 `D_m=2.82 h`、`D_c=0.28 h`。

复现时需冻结两个 manifest，并记录 base checkpoint、seed、clip hash 与 rollout 结果。`stratify_motions.py` 已能从五次 rollout 的结果生成 manifest 和带 hash 的审计报告；自动执行五次仿真评测仍是后续工作。

### 2.3 PACE

Stage II 固定 80% acquisition 环境从 `D_c` 自适应采样，20% consolidation 环境从 `D_m` 均匀采样。当前策略 `pi_theta` 与冻结参考策略 `pi_ref` 都由 Stage-I checkpoint 初始化。

Acquisition 使用 PPO；consolidation 使用确定性动作约束：

```text
L_con = E[||a_theta(s) - a_ref(s)||_2^2]
L = L_acq + lambda_con(t) * L_con

rho(t) = N_A(t) / (N_A(t) + N_C(t))
rho_bar(t) = beta * rho_bar(t-1) + (1-beta) * rho(t)
lambda_con(t) = min(1, lambda_base + kappa * max(0, rho_bar(t)-rho_ref))
```

固定值：`lambda_base=0.3`、`kappa=5`、`rho_ref=0.6`、`beta=0.99`。论文未定义 vectorized auto-reset 下的 valid sample；本实现定义为 rollout 中 `done == false` 的 transition，并写入运行契约。

### 2.4 STAR

challenging set 有 `B` 个时间 bin，采样概率为 `p_b`。transition 难度权重为 `w_t=B*p[b_t]`；`w_t>1` 属于高难度组 `H`，其余属于 `E`，两组 raw advantage 分别归一化。

轨迹片段是单环境 rollout 中被 termination 切开的最大连续序列。每个高难度 bin 独立按片段 raw advantage 均值排序，保留 top `max(ceil(0.05*n_b),1)`；入选片段组成 STAR pool。每个 PPO mini-batch 的 25% transition 从 pool 按片段平均难度加权重采样，其余正常随机采样。

## 3. 实现结构

```text
motion dataset / mastered + challenging manifests
  -> MotionCommand：固定环境角色与自适应 bin 采样
  -> RolloutStorage：role、bin、difficulty metadata
  -> PACEPPO.compute_returns：H/E 优势归一化与 STAR pool
  -> PACEPPO.update：acquisition PPO + consolidation action MSE
  -> checkpoint + run contract
```

| 模块 | 单一职责 |
| --- | --- |
| `extreme_rgmt/mdp/commands.py` | 数据集合并、角色固定、采样与 transition metadata |
| `rsl_rl/storage/rollout_storage.py` | 保存可选 PACE/STAR metadata，不影响标准 PPO |
| `rsl_rl/pace_star.py` | 可独立单测的权重、分组归一化、fragment 与 pool 函数 |
| `rsl_rl/algorithms/pace_ppo.py` | PACE loss、进度权重与 mixed mini-batch |
| `rsl_rl/runners/on_policy_runner.py` | metadata 接线和冻结 `pi_ref` 的 checkpoint 初始化 |
| `extreme_rgmt/agents/rsl_rl_ppo_cfg.py` | 两阶段 runner 配置 |
| `scripts/rsl_rl/extreme_rgmt/train.py` | 两阶段输入检查、训练与可审计运行契约 |
| `scripts/rsl_rl/extreme_rgmt/stratify_motions.py` | 将五次 rollout 结果冻结为 manifest 和审计报告 |

## 4. 当前状态与后续里程碑

已完成：

- 独立任务包、两阶段 Gym 注册与统一训练入口；
- mastered/challenging 数据契约及固定 acquisition/consolidation role；
- PACE 冻结参考动作 MSE 与自适应 `lambda_con`；
- STAR H/E 优势归一化、逐 bin top-5% fragment pool、25% 重采样；
- checkpoint 初始化、motion stratification、方法/数据/checkpoint hash 记录和纯 PyTorch 单测。

下一阶段按以下顺序推进，避免把未公开细节与已验证部分耦合：

1. 实现论文 Stage-I 10 帧 history、21-token reference cross-attention 和可配置 FSQ；
2. 将 action/数据契约升级为论文 29-DoF residual target，并给旧数据提供显式转换器；
3. 实现自动五次 rollout 的仿真评测，并接到现有 stratification 工具；
4. 运行 2--10 iteration Isaac Lab smoke、短程 learnability 和消融；
5. 补齐 MuJoCo 五 seed 评测后再进行论文数值对比。

## 5. 使用方法

Stage I：

```bash
python3 scripts/rsl_rl/extreme_rgmt/train.py \
  --stage base \
  --motion /path/to/all_motions.npz \
  --num_envs 8192 --device cuda:0
```

Stage II：

```bash
python3 scripts/rsl_rl/extreme_rgmt/train.py \
  --stage expansion \
  --mastered-motion /path/to/mastered.lst \
  --challenging-motion /path/to/challenging.lst \
  --base-checkpoint /path/to/model_stage1.pt \
  --num_envs 8192 --device cuda:0
```

Stage II 至少需要两个环境。入口禁止隐式 resume；它仅接受显式 Stage-I checkpoint，并以 `load_optimizer=False` 初始化当前策略和冻结参考策略。

## 6. 验收与未知项

当前代码验收：专项 unit tests、静态编译、格式检查、无旧 Teacher task/脚本/注册残留。完整论文复现还要求 Stage-I exact architecture、作者等价数据与多 seed MuJoCo 结果。

论文无法唯一确定：FSQ levels、attention heads/layers/activation、自适应采样器 EMA 与 clip、valid sample 精确定义、动作重定向/接触修正、29-DoF action scale 与 PD、Stage-II 总迭代数和 checkpoint 选择规则。仓库选择必须标为 `reproduction choice`，不能冒充论文配置。

## 7. 来源

- 论文 PDF：https://arxiv.org/pdf/2607.20110
- 项目页：https://zeonsunlightyu.github.io/Extreme-RGMT.github.io/
- PPO：https://arxiv.org/abs/1707.06347
- GAE：https://arxiv.org/abs/1506.02438
- FSQ：https://arxiv.org/abs/2309.15505

# Extreme-RGMT 论文调研与 UniTracker Lab 复现计划

## 1. 复现结论与边界

本文对应：Yubiao Ma 等，*Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control*，arXiv:2607.20110v1，2026-07-22。

论文项目页截至 2026-08-24 只提供论文和视频，没有公开训练代码、模型、训练数据、动作重定向流水线或完整配置。因此本仓库采用“公式忠实、接口可验证、实验可追踪”的 clean-room 复现策略：

1. 保留现有 `Unitracker_Teacher-v0` Stage-I teacher，不改变其 checkpoint ABI。
2. 新增并行的 Extreme-RGMT Stage-II 任务和训练入口。
3. 忠实实现 PACE 的 acquisition/consolidation 双角色训练、冻结基座动作约束和进度自适应权重。
4. 忠实实现 STAR 的困难度条件优势归一化、按 bin 的 top-k 轨迹片段选择及混合重采样。
5. 用显式 mastered/challenging manifest 作为论文 motion stratification 的产物，避免在训练时隐式改变数据划分。

当前仓库与论文 Stage-I 网络存在重要差异：仓库基线是 23 维动作、789 维 privileged actor；论文是 29 维 reference-residual action，并使用 10 帧 proprioception/action history、21-token reference window、cross-attention 和 FSQ。因此本轮代码是论文 **Stage-II 方法在当前 G1 UniTracker teacher 基座上的复现**，不是论文最终硬件 policy 的逐权重复刻。若要复现表 VI 的数值，还必须后续补齐论文 Stage-I actor、29-DoF 数据与作者评测集。

## 2. 论文方法细节

### 2.1 控制接口

论文控制频率为 50 Hz，底层 PD 为 500 Hz。actor 输入为：

- 10 帧 proprioception history：投影重力、base 角速度、相对默认位姿的关节位置、关节速度；
- 对应的 10 帧 previous-action history；
- 以当前时刻为中心的 21-token reference window；每个 token 包含 reference base 线速度、角速度、重力方向和关节位姿；
- critic 额外使用 reference base 高度、body link 位置/朝向和 robot base 线速度等 privileged state。

策略输出 29 维 residual joint position：

```text
q_target(t) = q_ref(t) + a(t)
tau(t) = Kp * (q_target(t) - q(t)) - Kd * qdot(t)
```

### 2.2 Stage I：generalist base policy

Stage I 在完整多源动作集上训练 `pi_base`。其 command encoder 包含：

1. proprioception 与 action 分别经 MLP 和 LayerNorm 编码；
2. 两类 token 交错排列后进入 causal history encoder；
3. reference token 单独经 MLP、LayerNorm 和位置编码；
4. history embedding 作为 query，reference window 作为 key/value 做 cross-attention；
5. 聚合结果分成两个 32 维 token，经 FSQ 离散瓶颈后与当前 proprioception、previous action 一起输入 actor。

论文超参数：

| 项目 | 取值 |
| --- | --- |
| state encoder | `[64, 128, 64]` |
| action encoder | `[29, 64, 64]` |
| command encoder | `[38, 128, 64]` |
| actor | `[1024, 1024, 512, 256]` |
| critic | `[1024, 1024, 512, 512]` |
| rollout horizon | 24 |
| PPO epochs / mini-batches | 5 / 4 |
| learning rate | `1e-3`, adaptive KL |
| target KL | `0.01` |
| gamma / GAE lambda | `0.99 / 0.95` |
| PPO clip | `0.2` |
| entropy coefficient | `0.005` |

论文训练动作共 3.096 h：LAFAN1 2.444 h、AMASS 0.511 h、in-house Xsens 0.141 h；全部重定向到 Unitree G1 并重采样为 50 Hz。

### 2.3 Motion stratification

Stage I 完成后：

1. 超过 10 s 的 motion 切成 10 s clip；短 motion 保持完整。
2. 每个 clip 做 5 次随机 rollout。
3. completion rate `>= 0.8` 进入 mastered set `D_m`，其余进入 challenging set `D_c`。
4. 论文最终得到 `D_m=2.82 h`、`D_c=0.28 h`。

划分结果必须冻结为两个 manifest，并记录 base checkpoint、随机种子、clip hash 与 5 次 rollout 结果。否则 Stage II 之间的数据比较不可复现。

### 2.4 PACE

Stage II 将环境固定分为两个角色：80% acquisition 环境从 `D_c` 自适应采样，20% consolidation 环境从 `D_m` 均匀采样。`pi_theta` 与冻结的 `pi_ref` 都由 `pi_base` 初始化。

Acquisition 环境使用 clipped PPO loss；consolidation 环境不用环境 reward 更新 actor，而约束当前确定性动作与冻结参考动作：

```text
L_con = E_s~D_m [ ||a_theta(s) - a_ref(s)||_2^2 ]
L = L_acq + lambda_con(t) * L_con
```

论文用有效 acquisition sample 的实现比例衡量训练进度：

```text
rho(t) = N_A(t) / (N_A(t) + N_C(t))
rho_bar(t) = beta * rho_bar(t-1) + (1-beta) * rho(t)
lambda_con(t) = min(1, lambda_base + kappa * max(0, rho_bar(t)-rho_ref))
```

固定值为 `lambda_base=0.3`、`kappa=5.0`、`rho_ref=0.6`、`beta=0.99`。论文没有公开“valid sample”在 vectorized auto-reset 环境中的具体掩码。本复现将其定义为 rollout 内未触发 done 的 transition；该定义会写入 run config 和日志。

### 2.5 STAR

对 challenging set 的 `B` 个时间 bin，自适应采样给出概率 `p_b`。transition `t` 的困难度权重为：

```text
w_t = B * p[b_t]
```

`w_t > 1` 为高困难组 `H`，其余为 `E`。先保存 raw advantage `A_raw=R-V`，再对 `H/E` 分组独立归一化，归一化结果用于 PPO。

轨迹片段定义为单个环境在当前 rollout 内、被 termination 边界切开的最大连续 transition 序列。对每个高困难 bin `b` 和片段 `tau`，计算：

```text
q[b,tau] = mean(A_raw(t)), t belongs to H, bin(t)=b, fragment(t)=tau
```

每个 bin 独立保留 top `max(ceil(0.05*n_b), 1)` 个 pair；入选片段的所有 acquisition transition 构成 pool `P`。每个 PPO mini-batch 有 25% transition 从 `P` 重采样，其余走标准随机采样。pool 内 transition 的概率与所在片段的平均 `w_t` 成正比。

## 3. 仓库映射

```text
mastered/challenging .lst
  -> MotionCommand role-aware RSI
  -> transition metadata (role, adaptive bin, difficulty)
  -> RolloutStorage
  -> PACEPPO.compute_returns(): STAR H/E normalization + pool construction
  -> PACEPPO.update(): acquisition PPO + consolidation action MSE
  -> model checkpoint
```

计划中的文件职责：

| 文件 | 职责 |
| --- | --- |
| `mdp/commands.py` | 合并两个数据集、固定 env role、mastered 均匀采样、challenging adaptive sampling、导出 transition metadata |
| `rsl_rl/storage/rollout_storage.py` | 仅新增 3 个可选 tensor，保存 acquisition mask、bin id、difficulty weight；标准 PPO 不使用 |
| `rsl_rl/algorithms/pace_ppo.py` | PACE loss、进度权重、STAR pool 和 mixed mini-batch |
| `on_policy_runner.py` | action 前抓取当前 reference-bin metadata；checkpoint load 后冻结 `pi_ref` |
| `agents/extreme_rgmt_ppo_cfg.py` | 论文 Stage-II 超参数和独立实验名 |
| `train_extreme_rgmt.py` | 强制要求 base checkpoint 与两个 manifest，写入数据和方法 contract |
| `stratify_motions.py` | 从五次 rollout 结果生成冻结 manifest 与审计报告 |

## 4. 实施里程碑

当前分支的第一批最小实现已覆盖 M1-M3 的静态链路。Stage-II policy 保持现有 Stage-I teacher 的 `[512, 512, 256, 128]` actor/critic 形状，以保证 base checkpoint 可加载；论文的 `[1024, ...]` 网络只记录为后续 Stage-I 复现目标，不混入本批 PACE/STAR 改动。

### M1：数据与角色契约

- 两个 manifest 非空、不能包含相同文件。
- env 数量乘 acquisition fraction 后，两个角色都至少有一个环境。
- acquisition 环境永远只采样 `D_c`；consolidation 环境永远只采样 `D_m`。
- consolidation 的 difficulty 固定为 1，不进入 STAR。

### M2：PACE/STAR 纯 PyTorch 实现

- H/E 独立归一化在单元素、空组和零方差时无 NaN。
- fragment ID 不跨 env、不跨 done。
- top-k 是逐 bin 而不是全局。
- pool 为空时严格退化成 acquisition PPO。
- reference policy `requires_grad=False` 且不进入 optimizer/checkpoint 的可训练参数。

### M3：训练接线

- Stage-II 只允许显式 base checkpoint 启动。
- checkpoint load 后同步 `pi_theta` 和 `pi_ref`，不恢复 Stage-I optimizer。
- 日志至少记录 `rho`、`rho_bar`、`lambda_con`、STAR pool fraction、H fraction、acquisition/consolidation loss。

### M4：运行验证

1. 纯 Python unit tests。
2. Isaac Lab registry/config construction。
3. 1 env reset/step 不适用，因为 PACE 至少需要两个角色；使用 2 env。
4. 16-64 env、2-10 iteration PPO smoke。
5. 小数据 learnability，之后再做完整训练。

## 5. 训练与评测协议

Stage I 基座先由现有 teacher 入口训练。Stage II 示例：

```bash
python3 scripts/rsl_rl/extreme_rgmt/train_extreme_rgmt.py \
  --mastered-motion /path/to/mastered.lst \
  --challenging-motion /path/to/challenging.lst \
  --base-checkpoint /path/to/model_stage1.pt \
  --num_envs 8192 --device cuda:0
```

最小消融矩阵：

| 实验 | consolidation role | `L_con` | adaptive lambda | STAR |
| --- | --- | --- | --- | --- |
| fine-tune | 否 | 否 | 否 | 否 |
| w/o `L_con` | 是 | 否 | 否 | 是 |
| fixed PACE | 是 | 是，0.3 | 否 | 是 |
| PACE w/o STAR | 是 | 是 | 是 | 否 |
| full | 是 | 是 | 是 | 是 |

评测与论文一致时需要 MuJoCo 侧 5 seeds，并报告 completion、root-relative MPJPE、joint velocity error 和 joint acceleration error。失败条件是 robot root height 相对 reference 偏差超过 0.2 m。必须分别报告 mastered/generalist、challenging、unseen 三组，不能只汇总一个均值。

## 6. 无法由论文唯一确定的内容

- FSQ 每一维的 level 数、attention head 数、causal encoder 层数与 activation。
- adaptive sampler 的 EMA 系数、clip 上限和完整失败统计定义。
- valid sample 的精确实现。
- motion retargeting、接触修正、Xsens 清洗和 unseen set。
- 29-DoF action scale、PD 数值和硬件 safety envelope。
- Stage-II 总迭代数、batch size 跨 GPU 语义和 checkpoint 选择规则。

这些项目不得靠猜测标成“论文配置”。本仓库采用的值必须在 resolved config 中标记为 reproduction choice，并通过消融验证。

## 7. 验收标准

- 现有 Stage-I tests 全部通过，旧 task 行为不变。
- 新增 PACE/STAR unit tests 全部通过。
- clean checkout 能从两个 manifest 和 base checkpoint 构造 Stage-II run。
- 每个 run 保存 git diff、agent/env YAML、两个 resolved manifest、所有 clip hash、base checkpoint hash 和 Extreme-RGMT method contract。
- 完整复现声明只在 Stage-I architecture、数据集与 MuJoCo 评测也补齐，并得到多 seed 结果后成立。

## 8. 来源

- 论文 PDF：https://arxiv.org/pdf/2607.20110
- 项目页：https://zeonsunlightyu.github.io/Extreme-RGMT.github.io/
- PPO：https://arxiv.org/abs/1707.06347
- GAE：https://arxiv.org/abs/1506.02438
- FSQ：https://arxiv.org/abs/2309.15505

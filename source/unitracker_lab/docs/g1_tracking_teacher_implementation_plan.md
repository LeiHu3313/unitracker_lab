# G1 UniTracker Stage-1 Tracking Teacher 实现计划

## 0. 结论

当前仓库只保留一条 G1 Teacher 链路：

```text
prepared G1 reference motion
  -> G1MotionCommand（RSI、严格的 t -> t+1 时序）
  -> 16-body / 23-joint privileged oracle observation
  -> RSL-RL PPO Teacher
  -> 23-D joint-position target
  -> G1 implicit PD
  -> whole-body tracking reward
  -> checkpoint / play / evaluation
```

正式 Gym ID：

```text
Unitracker_Teacher-v0
Unitracker_Teacher-Play-v0
```

`Unitracker_Teacher-v0` 从此只表示 G1 UniTracker Stage-1 Teacher。现有旧机器人任务、资产、验证脚本和文档假设先删除或替换，不保留旧兼容层。第一版按 UniTracker Stage-1 的语义实现：无 CVAE、无历史编码、无学生、无 DAgger、无外力/PD/延迟等强 dynamics DR。

本文不讨论 AMASS 清洗、PHC filtering 和 Human-to-G1 retargeting 的实现；它们的输出被视作本任务的输入边界。

## 1. 参考对象与优先级

### 1.1 语义依据

任务语义以本次给出的 UniTracker Stage-1 范式为最高优先级：

- 29-DoF G1，固定左右腕部各 3 个关节，策略控制 23 DoF。
- Teacher Actor 使用当前完整 privileged whole-body state。
- Goal 使用下一控制时刻 reference 与当前 robot 的显式误差。
- 所有空间量 canonicalize 到当前 robot root local frame。
- PPO 输出 joint-position target，由底层 PD 执行。
- 使用 RSI、whole-body tracking reward、固定正则化权重和 early termination。
- Teacher 只使用 asset-property DR，不使用环境动力学扰动。

### 1.2 工程参考

经典 G1 tracking 参考仓库：

```text
/home/hul/whole_body_tracking
commit: cd65172032893724b445448818c34165846d847d
```

重点参考：

| 能力 | 参考文件 |
| --- | --- |
| G1 URDF 导入、actuator/PD、action scale | `whole_body_tracking/robots/g1.py` |
| G1 14-body 映射与配置覆盖 | `tasks/tracking/config/g1/flat_env_cfg.py` |
| Motion loader、RSI、phase sampling | `tasks/tracking/mdp/commands.py` |
| local-frame observation | `tasks/tracking/mdp/observations.py` |
| whole-body exponential reward | `tasks/tracking/mdp/rewards.py` |
| tracking termination | `tasks/tracking/mdp/terminations.py` |
| Gym 注册、PPO、train/play/export | `config/g1/`、`scripts/rsl_rl/`、`utils/exporter.py` |

该仓库提供实现模式，不提供本任务最终语义。以下内容不能照搬：

| BeyondMimic 当前行为 | 本任务目标 |
| --- | --- |
| 策略控制全部 29 DoF | 固定 6 个 wrist，控制 23 DoF |
| Actor 是相对紧凑的 tracking observation | Actor 是完整 privileged oracle + next-frame error |
| raw current reference `q/qdot` command | 显式 `reference[t+1] - robot[t]` goal |
| 默认 interval velocity push | Stage-1 禁止 push |
| 自适应失败 phase sampling | 第一条基线使用均匀 RSI；自适应采样只做后续 ablation |
| W&B Registry 是训练硬依赖 | 本项目使用 CLI 本地路径，W&B 只能作为可选 logger |
| Isaac Sim 4.5 / Isaac Lab 2.1 | 本项目固定 Isaac Sim 5.1 / Isaac Lab 2.3.2 |

## 2. 范围与非目标

### 2.1 本轮必须完成

- G1 29-DoF 资产闭包及 `ArticulationCfg`。
- 明确、可测试的 23 个控制关节与 6 个固定 wrist 关节。
- 16-body tracking map。
- 单 NPZ、递归目录或 `.txt/.lst` manifest 的严格加载、重排、多 motion RSI 与 `t -> t+1` 时序。
- 789-D Teacher Actor observation 和同构 Critic observation。
- 23-D position target action + implicit PD。
- 论文范式对应的 tracking reward、固定正则化权重和 early termination。
- asset-only DR。
- 唯一的 G1 Gym registration、PPO cfg、train/play、validator。
- 纯 Python contract tests、Isaac Sim reset/step tests、PPO smoke test。

实施第一步先清理当前仓库的旧机器人实现：

- 重建 `tasks/manager_based/unitracker_teacher/`，只留下 G1 task、G1 MDP 和 G1 agent 配置。
- `assets/` 删除其他机器人资产，只保留 G1 Teacher 运行所需资产以及与任务无关的项目公共资源。
- 重写现有 `scripts/rsl_rl/teacher/train_teacher.py`、`train_teacher.sh` 和 `validate_motion_npz.py`，使其只接受 G1 contract。
- 删除只描述旧机器人训练路线的实现文档，避免出现两个相互冲突的 Teacher 定义。
- 不提供旧 checkpoint、旧 motion schema 或旧 Gym ID 语义的兼容适配。

### 2.2 后续能力，但预留接口

- 超大数据集的 memmap/cache。
- 8192-env 生产训练与多 GPU。
- 全数据集评估、失败 phase 报告和动作级指标。
- ONNX/JIT 导出以及 Stage-2 DAgger 所需的 teacher-action 接口。

### 2.3 明确不进入 Stage-1 Teacher

- CVAE、latent sampling。
- history encoder、proprioception history。
- Student、DAgger、BetaMix、Residual PPO。
- torso push、random force、torque noise。
- PD gain randomization、actuator delay。
- rough terrain curriculum。
- 面向部署的 observation noise。

## 3. 目录与模块边界

清理后目标结构：

```text
source/unitracker_lab/unitracker_lab/
├── assets/g1/
│   ├── __init__.py
│   ├── g1.py                         # ArticulationCfg、PD、joint/body contracts
│   ├── g1_29dof_rev_1_0.urdf
│   ├── meshes/
│   ├── asset_manifest.json           # 来源、hash、导入选项、许可证信息
│   └── README.md
└── tasks/manager_based/unitracker_teacher/
    ├── __init__.py                   # 注册唯一的 Train/Play task
    ├── contracts.py                  # 23-joint、16-body、obs/action/timing 常量
    ├── motion_schema.py              # strict NPZ schema + dataset resolver
    ├── unitracker_teacher_env_cfg.py
    ├── mdp/
    │   ├── __init__.py
    │   ├── commands.py               # G1 motion schema、RSI、时序
    │   ├── observations.py           # 789-D oracle
    │   ├── rewards.py
    │   ├── terminations.py
    │   └── curriculum.py
    └── agents/
        ├── __init__.py
        └── rsl_rl_ppo_cfg.py

scripts/rsl_rl/teacher/
├── train_teacher.py
├── train_teacher.sh
├── play_teacher.py
└── validate_motion_npz.py

tests/unitracker_teacher/
├── test_contracts.py
├── test_motion_schema.py
├── test_oracle_layout.py
├── test_reward_curriculum.py
├── test_registry.py
├── test_env_reset_step.py
└── test_temporal_alignment.py
```

`unitracker_teacher/mdp/` 直接实现 G1 语义。旧的 joint/body/order 假设整体删除，不设计机器人选择开关或抽象基类；这里只有一个机器人，配置和错误信息都可以明确写成 G1。

同时更新：

```text
source/unitracker_lab/MANIFEST.in
```

manifest 清理为只打包 G1 Teacher 所需的 `*.urdf`、`*.STL`、`*.json` 和 `*.md`；删除不再使用的旧资产规则。

## 4. G1 跨模块硬契约

### 4.1 物理模型与 DoF

使用已经复制到项目中的：

```text
assets/g1/g1_29dof_rev_1_0.urdf
sha256: 8df048597b758a4f868c1eef12ba995e331420a5aceef810a07c12e3b208ac13
```

不直接换成外部目录里的 `g1_23dof_rev_1_0.urdf`。后者的 23 个 movable joints 实际是：

- 固定 `waist_roll_joint`、`waist_pitch_joint`；
- 固定左右 `wrist_pitch_joint`、`wrist_yaw_joint`；
- 保留左右 `wrist_roll_joint`。

这与本任务“固定左右腕部各 3 DoF、保留三自由度 waist”的定义不同。两者虽然都是 23 维，但 checkpoint 的每一维物理意义不同。

### 4.2 23 个控制关节顺序

该顺序同时约束 policy action、joint observation、reference selection、reward、checkpoint、评估和未来部署：

```python
G1_CONTROLLED_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
```

固定关节：

```python
G1_LOCKED_WRIST_JOINT_NAMES = [
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

第一版 lock position 全部为 `0.0 rad`。这些关节仍存在于 29-DoF articulation 中，但：

- 不进入 policy action；
- 不进入 `q/qdot/previous_action` observation；
- reset 时强制写入 lock position 和零速度；
- 独立 implicit actuator group 持续把它们保持在 lock position；
- 测试必须证明任意 23-D action 不会改变 wrist target。

如果 Isaac Lab 的 implicit target buffer 无法可靠保持未进入 ActionTerm 的 wrist，则生成一个明确的 `g1_29dof_wrist_locked.urdf` 或零维 fixed-target term；不能让腕部变成无驱动的被动关节。

### 4.3 16-body tracking 顺序

16-body map 采用当前 G1 retarget 配置的 hips/spine/chest/head/limbs 语义，并吸收 `whole_body_tracking` 的 pelvis/limbs 选择方式：

```python
G1_TRACKING_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_elbow_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_rubber_hand",
]
```

附加集合：

```python
G1_ROOT_BODY_NAME = "pelvis"
G1_FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
G1_HAND_BODY_NAMES = ["left_rubber_hand", "right_rubber_hand"]
```

本 URDF 的 `head_link` 通过 fixed joint 连接，Isaac Lab 2.3.2 的 URDF converter 默认 `merge_fixed_joints=True`。为了确保 16-body contract 可解析，第一版设置 `merge_fixed_joints=False`，并在一环境启动测试中打印和冻结 live articulation body order。若性能不理想，再生成只保留所需 fixed links 的兼容 USD；不能静默删掉 `head_link` 后仍声称是 16-body observation。

### 4.4 时间和形状

```python
NUM_PHYSICAL_JOINTS = 29
NUM_CONTROLLED_JOINTS = 23
NUM_LOCKED_WRISTS = 6
NUM_TRACKING_BODIES = 16
NUM_ACTIONS = 23
SIM_DT = 0.005
DECIMATION = 4
CONTROL_DT = 0.020
CONTROL_HZ = 50
TEACHER_OBS_DIM = 789
CRITIC_OBS_DIM = 789
```

生产配置目标为 8192 environments；所有功能验证必须允许通过 CLI 覆盖为 1、16、64 或 4096。

## 5. G1 资产、PD 与 action

### 5.1 资产配置

以 `whole_body_tracking/robots/g1.py` 为 G1 参数起点，适配当前 `g1_29dof_rev_1_0.urdf` 的 joint names 和 limits：

- floating base；
- cylinders 转 capsules；
- contact sensors；
- self collision；
- PhysX position/velocity solver iterations = 8/4；
- initial root height 约 0.76 m；
- `soft_joint_pos_limit_factor=0.9`；
- URDF importer drive gains 清零，由 Isaac Lab actuator cfg 提供 PD。

actuator groups：

```text
legs          hips + knees
feet          ankle pitch/roll
waist         waist yaw/roll/pitch
arms          shoulder 3-DoF + elbow
locked_wrists wrist roll/pitch/yaw, fixed target only
```

PD stiffness/damping/armature 先采用 BeyondMimic G1 的 10 Hz natural-frequency 配置作为可运行基线，但必须写成 G1 资产常量并保存到 resolved `env.yaml`，不能运行时从外部仓库导入。effort/velocity limits 以当前 URDF 与经过确认的 actuator override 为准；二者不一致时在资产测试中显式报告。

### 5.2 23-D action

```text
normalized action a_t[23]
  -> optional clip to [-1, 1]
  -> q_target = q_default_controlled + action_scale * a_t
  -> implicit PD
```

`JointPositionActionCfg` 必须使用显式的 `G1_CONTROLLED_JOINT_NAMES`、`preserve_order=True` 和 `use_default_offset=True`。

action scale 初始采用经典项目的 torque-aware 规则：

```text
scale[j] = 0.25 * effort_limit[j] / stiffness[j]
```

启动时把解析后的 23 个 action scales 连同 joint order 写入日志。完成单 motion overfit 后，再与统一 `0.25 rad` 做一次 ablation；不要在首轮同时调 reward、PD 和 action scale。

## 6. Prepared motion 输入边界

本计划不实现 retarget，但训练 loader 必须严格定义其输出契约。

### 6.1 NPZ 字段

```text
fps
joint_names
body_names
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
```

约定：

- meters、radians、seconds；Z-up、X-forward；
- quaternion 内存格式为 WXYZ；
- 数组首维都是 frame；
- `joint_names` 必须存在，loader 按名字重排，禁止仅凭 shape 推断顺序；
- `body_names` 必须存在，loader 按 16-body contract 重排，禁止使用 live articulation index 直接索引 NPZ；
- `joint_pos/joint_vel` 可包含完整 29 DoF，也可只包含受控 23 DoF；
- 若包含 29 DoF，六个 wrist 的 reference position 必须在 lock tolerance 内、velocity 接近零；
- reference 至少包含两个可用 control frames；
- position/velocity 必须 finite，quaternion norm 必须在 tolerance 内。

### 6.2 reference FPS

第一版强制 `fps == 50`，使“下一帧”严格等于“下一 control step”。不能像经典项目一样无条件 `time_steps += 1` 却接受任意 FPS，也不能用 `round(fps * control_dt)` 丢失 30 Hz/60 Hz 的相位。

后续若必须接受非 50 Hz reference，则增加连续时间 sampler：位置/速度线性插值、orientation SLERP，并把 goal 定义为 `reference(t + CONTROL_DT)`；它应作为单独能力实现和测试。

### 6.3 单 motion 到生产数据集

实施顺序：

1. 单 NPZ，uniform RSI，用于逻辑与 learnability。已实现。
2. 本地 `.txt/.lst` manifest 或递归 directory，多 motion uniform sampling。已实现。
3. memmap/cache，用于无法常驻 CPU/GPU 内存的超大训练集。
4. 可选 failure-aware phase sampling。

自适应 phase sampling 可以参考 `whole_body_tracking` 的 failed-bin kernel，但不进入首条基线，因为 UniTracker Stage-1 范式只要求 RSI。无论后续使用哪种 sampler，run 目录必须保存实际 motion 清单和 sampling 配置。

当前实现启动时逐条严格校验，再把同字段数组拼接为连续 tensor，并另外保存每条 clip 的
`start/length`。采样顺序是“均匀选 motion，再在该 motion 的 `[0, F-2]` 内均匀选 RSI phase”，
因此短轨迹不会被长轨迹压低采样概率，`k -> k+1` 也不会穿过 clip 边界。

对于 `xyz + quaternion(WXYZ) + 29 joint qpos` 的 G1 数据，可使用独立转换脚本：

```bash
python scripts/rsl_rl/teacher/convert_g1_qpos_dataset.py \
  /path/to/qpos_dataset \
  /path/to/prepared_teacher_dataset
```

转换器将 root quaternion 做 SLERP、其他自由度线性重采样到 50 Hz，锁定 6 个 wrist，随后按当前
Teacher URDF 做 FK，并由重采样后的轨迹计算 joint/body 速度。输出包含每条 motion 的 NPZ、
`motions.lst` 和带源路径/hash 的 `conversion_manifest.json`。

## 7. `t -> t+1` 时序契约

这是实现中最容易出现隐性 off-by-one 的部分，必须先写测试再写 reward。

### 7.1 每个正常 control step

```text
command.phase_index  = k
command.target_index = k + 1

observation:
  current robot state s_k
  target reference ref[k+1]

policy:
  a_k = pi_oracle(s_k, root/body ref[k+1], joint ref[k:k+5])

physics:
  apply a_k for 4 x 0.005 s

reward / termination:
  compare resulting robot state s_{k+1} with ref[k+1]

command update:
  phase_index  <- k+1
  target_index <- k+2
```

命令对象必须暴露明确命名的属性：

```text
current_ref_*
target_ref_*
phase_index
target_index
```

reward 不得通过含义模糊的 `command.body_pos_w` 猜测当前还是下一帧。

### 7.2 RSI reset

reset 时：

1. uniform sample `k in [0, F-2]`；
2. robot floating root 写入 `ref[k]`；
3. controlled joints 写入 `ref[k].q/qdot`；
4. locked wrists 写入固定角和零速度；
5. observation 指向 `ref[k+1]`。

第一条基线不叠加 root pose、velocity 或 joint noise。若后续需要 recovery 扰动，作为单独 ablation 打开。

Isaac Lab 在 terminated env reset 后，会在同一个 `env.step()` 末尾再次调用 `command_manager.compute()`。因此 `G1MotionCommand` 需要 `just_reset` mask：刚 RSI 的 env 在这次 `_update_command()` 中跳过 phase promotion，否则 reset 后第一次 observation 会错误地变成“robot 在 k，goal 在 k+2”。

### 7.3 motion end

当 `target_index == F-1` 时，仍先完成该动作、计算最后一帧 reward，然后用一个 `time_out=True` 的 `motion_end` term reset。motion end 不算 tracking failure，也不触发 `-200` early-termination penalty。

## 8. Teacher oracle observation

### 8.1 坐标规范

canonical frame 使用当前 robot `pelvis`：

- origin：当前 robot pelvis position；
- rotation：当前 robot pelvis full orientation，而不是只取 yaw；
- body position：减 pelvis position 后旋转到 pelvis frame；
- linear/angular velocity：只旋转到 pelvis frame；
- body orientation：`R_root^T R_body`；
- reference body pose goal：robot/reference 分别变换到各自 pelvis frame 后再求误差；
- reference velocity error：先 world-space 相减，再旋转到当前 pelvis frame；
- quaternion 内部使用 WXYZ，送入网络前转换为 rotation 6D。

所有 observation term 都不加噪声、不加 history。网络 normalization 是否开启由 runner cfg 明确控制。

### 8.2 Actor 精确布局

采用一个单独的 pelvis root、15 个非 root tracking bodies、29 physical joints、rotation 6D。所有 body
block 都排除 pelvis，避免将恒为零的 root-local position 和恒为 identity 的 root-local rotation
送入网络。全局 root x/y 和绝对 yaw 不进入 observation ABI。

| 顺序 | block | 维度 |
| ---: | --- | ---: |
| 1 | current root height above environment origin | `1` |
| 2 | current projected gravity in root frame | `3` |
| 3 | current root linear/angular velocity in root frame | `3 + 3` |
| 4 | current non-root body position/orientation in root frame | `15 x 3 + 15 x 6 = 135` |
| 5 | current non-root body linear/angular velocity in root frame | `15 x 3 + 15 x 3 = 90` |
| 6 | current all-joint position relative to default / velocity | `29 + 29 = 58` |
| 7 | current left/right foot binary contact and previous action | `2 + 23 = 25` |
| 8 | target root height, orientation, linear/angular velocity error | `1 + 6 + 3 + 3 = 13` |
| 9 | target torso global-position error in current root frame | `3` |
| 10 | target non-root body position/orientation error | `15 x 3 + 15 x 6 = 135` |
| 11 | target non-root body linear/angular velocity error | `15 x 3 + 15 x 3 = 90` |
| 12 | reference controlled joint command at `t, ..., t+4` | `5 x (23 + 23) = 230` |
| | **总计** | **789** |

其中：

```text
current oracle state          = 318
next-frame root/body goal     = 241
five-frame joint command      = 230
teacher actor input            = 789
```

joint command 使用 `t, t+1, t+2, t+3, t+4` 五个 reference frame，末尾帧按每个 clip 的末端重复；它不是 robot-state history。root/body tracking goal 仍然是显式的 `t -> t+1` 误差。

### 8.3 Critic

第一版 Critic 使用独立的 `critic` observation group，但布局与 Teacher Actor 完全相同，也是 789 维：

```python
obs_groups = {"policy": ["teacher"], "critic": ["critic"]}
```

保持两个 group 是为了未来可以做 asymmetric critic ablation；首条基线不额外给 Critic 时间索引或 terrain，双脚当前 contact mask 则作为 Teacher 的完整仿真状态输入。

### 8.4 observation contract tests

测试不能只断言总维度 789，还要对每个 block 做 slice test：

- identity pose 时 rot6d 的确切编码；
- global yaw/translation 同时施加给 robot/reference 后 local observation 不变；
- 单独改变一个 body angular velocity，只改变对应 3 维；
- 单独改变 `ref[k+1]`，只改变 next-frame goal 与相应 future-command frame；
- future command 的时间索引严格为 `k, ..., k+4`，并在 clip 尾部截断；
- wrist state 必须进入 29-DoF current joint blocks，但 goal 仍仅包含 23 个 controlled joints；
- 双脚 contact mask 的顺序与 `G1_FOOT_BODY_NAMES` 一致；
- block 顺序与导出 metadata 一致。

## 9. Reward、curriculum 与 termination

### 9.1 tracking reward

目标权重以“相对构型优先、torso 世界系软锚定”为准：

| term | weight | target |
| --- | ---: | --- |
| global torso position | `+0.5` | torso xyz in world frame |
| global torso orientation | `+0.5` | torso rotation, including yaw |
| global torso linear/angular velocity | `+0.5 / +0.5` | torso world-frame velocity |
| relative body position | `+1.0` | 15 non-root bodies in their own pelvis frame |
| relative body rotation | `+1.0` | 15 non-root body orientations relative to pelvis |
| controlled joint position | `+0.5` | 23 controlled joints |
| controlled joint velocity | `+0.5` | 23 controlled joints |
| body linear velocity | `+0.5` | 16-body target velocity |
| body angular velocity | `+0.5` | 16-body target angular velocity |

reward kernel 使用可配置 exponential tracking 形式：

```text
exp(-mean_squared_error / sigma^2)
```

首轮可运行默认值从 BeyondMimic 的量级出发：

| term | provisional sigma |
| --- | ---: |
| global torso position | 0.30 m |
| global torso orientation | 0.40 rad |
| global torso linear velocity | 1.00 m/s |
| global torso angular velocity | 2.50 rad/s |
| relative body position | 0.30 m |
| relative body orientation | 0.40 rad |
| joint position | 0.25 rad |
| joint velocity | 2.50 rad/s |
| body linear velocity | 1.00 m/s |
| body angular velocity | 2.50 rad/s |

Table I 给出了 weights，但未在当前需求中给出所有 kernel temperatures；因此 sigma 必须作为 resolved config 和实验 manifest 的显式字段。正式大训练前先用 one-motion overfit 冻结，不能把 provisional 数值当成论文已确认事实。

### 9.2 regularization 与 penalty

| term | full weight | 计算范围 |
| --- | ---: | --- |
| action rate | `-0.1` | 23-D normalized action difference |
| controlled joint velocity L2 | `-1e-4` | 23 controlled joints |
| controlled joint position limits | `-10.0` | 23 controlled joints' soft limits |
| foot slippage | `-1.0` | contact feet planar velocity |
| early termination | `-50` | fall/tracking failure only |

不在 Teacher v1 中加入 torque、undesired-contact、feet-air-time、survival 或 self-collision reward；它们必须作为后续单独 ablation。

Isaac Lab `RewardManager` 会自动把每个 reward 乘以 `step_dt=0.02`。early termination 是一次性 penalty，必须避免被缩小。实现方式：termination indicator term 返回 `terminated / step_dt`，配置 weight 为 `-50`；测试直接断言 failure step 的 penalty 等于 `-50`，timeout/motion-end 为 `0`。

### 9.3 reward curriculum

所有 tracking、regularization 与 early-termination 项从第 0 iteration 起就是完整权重；当前任务关闭 reward curriculum。

early-termination 条件和其 `-50` reward penalty 均从训练开始启用。若要重新做 curriculum ablation，可改用保留的 `*_curriculum` MDP term 和显式的 start/end iteration 参数。

### 9.4 early termination

非 timeout failure 只有两类：

```text
fall:
  max(abs(projected_gravity_root.x), abs(projected_gravity_root.y)) > 0.8

tracking_failure:
  mean_i ||target_body_pos[i] - robot_body_pos[i]||_2 > 0.5 m
```

第二项对 16 tracking bodies 取 mean。position distance 的范数对坐标旋转不敏感，但 target 与 robot 必须使用同一个 env origin 和同一个 `k+1` 时刻。

另外保留：

- episode length timeout：truncation，不罚 `-200`；
- motion end：truncation，不罚 `-200`。

## 10. Stage-1 asset-only DR

训练地形使用纯 plane。Teacher observation 不加噪声。

允许的 startup asset randomization：

| property | baseline range |
| --- | --- |
| static friction | `[0.3, 1.6]` |
| dynamic friction | `[0.3, 1.2]` |
| restitution | `[0.0, 0.5]` |
| torso/pelvis CoM x | `[-0.025, 0.025] m` |
| torso/pelvis CoM y/z | `[-0.05, 0.05] m` |
| non-fixed link mass scale | `[0.8, 1.2]` |

明确禁止出现在主配置中的 event/actuator 选项：

```text
push_by_setting_velocity
apply_external_force/impulse
randomize_actuator_gains
torque noise
actuator min/max delay
rough terrain
observation corruption
```

增加一个 DR whitelist test：枚举激活的 `EventTerm` 和 actuator delay，确保主任务没有上述 dynamics perturbation。Play 配置关闭全部 asset DR，使用 nominal asset。

## 11. PPO 配置

算法路径：

```text
RslRlVecEnvWrapper
  -> local rsl_rl OnPolicyRunner
  -> ActorCritic / PPO
  -> RolloutStorage
```

首条 baseline 从 `whole_body_tracking` 的 G1 PPO 配置起步：

| 参数 | 值 |
| --- | ---: |
| num steps per env | 24 |
| max iterations | 30,000（CLI 可覆盖） |
| save interval | 500 |
| actor hidden dims | `[512, 256, 128]` |
| critic hidden dims | `[512, 256, 128]` |
| activation | ELU |
| init action std | 1.0 |
| actor/critic obs normalization | True |
| PPO clip | 0.2 |
| entropy coef | 0.005 |
| epochs / mini-batches | 5 / 4 |
| learning rate | `1e-3`, adaptive KL |
| gamma / lambda | 0.99 / 0.95 |
| desired KL | 0.01 |
| max grad norm | 1.0 |

生产默认 `num_envs=8192`，但 runner 配置与 task 语义分开：环境数、network width 和总 iterations 都是算力/收敛参数，不是 observation/action contract。若 789-D oracle 在 `[512,256,128]` 下欠拟合，第二个受控实验再比较较大的 `[2048,1024,512]` 网络。

首版使用标准本地 `OnPolicyRunner`，不复制 BeyondMimic 的 W&B 专用 runner。训练必须确认 import 到当前 submodule 中的 RSL-RL，而不是系统安装版本。

## 12. Train、Play、日志与 checkpoint

### 12.1 训练入口

```bash
scripts/rsl_rl/teacher/train_teacher.sh \
  --motion /absolute/path/to/prepared_g1_dataset \
  --task Unitracker_Teacher-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --seed 42 \
  --headless
```

生产时再提高到 8192。脚本不内置个人数据路径、不默认 resume。
`--motion` 也可传单个 prepared NPZ，或每行一个 NPZ 路径的 `.txt/.lst` manifest；manifest 中的相对路径以 manifest 所在目录解析。

### 12.2 Play task

`Unitracker_Teacher-Play-v0` 与 Train 使用完全相同的 observation/action/network contract，只修改：

- `num_envs` 默认 1；
- 每个 env 按 `env_id % motion_count` 确定性选择 motion，并从该 clip 的 frame 0 开始；
- 关闭 asset DR；
- 关闭 RSI noise（训练本来也默认关闭）；
- 打开 reference/robot 16-body markers；
- episode 长度由 motion end 控制。

### 12.3 run 快照

```text
logs/rsl_rl/unitracker_teacher/<timestamp>_<run>/
├── model_*.pt
├── params/
│   ├── env.yaml
│   ├── agent.yaml
│   ├── g1_contract.json
│   ├── asset_manifest.json
│   ├── motion_data_resolved.json
│   ├── used_motions.txt
│   └── reference_project.json
└── videos/                         # optional
```

`g1_contract.json` 至少保存：

- 23 controlled joint names；
- 6 locked wrist names/positions；
- 16 body names；
- action scale、default joint position；
- observation block names、offsets、dimensions；
- quaternion/rotation convention；
- `sim_dt/decimation/control_dt`；
- reward weights/sigmas 与 curriculum 开关；
- URDF SHA-256。

checkpoint 本身沿用 RSL-RL 的 model/optimizer/iteration 结构。Stage-2 只读取 Teacher Actor 与 observation contract；不会把 Critic 或 optimizer 当成 action label source。

## 13. 分阶段实施与验收

### M0：清理旧实现、建立 G1 资产和静态契约

实现：

- 删除当前 `unitracker_teacher/` 中的旧任务实现并按目标结构重建；
- 删除旧机器人资产、验证逻辑和只服务于旧任务的文档；
- 确认仓库中只注册 `Unitracker_Teacher-v0` 与 `Unitracker_Teacher-Play-v0`；
- G1 package data 与 manifest；
- `g1.py`、23/6 joint lists、16 body list；
- actuator groups、PD、action scale；
- contract tests。

验收：

- wheel/editable install 后 URDF 与 35 个 mesh 都可访问；
- URDF XML valid、hash 与 manifest 一致；
- controlled/locked 两集合无交集，合并后严格等于 live 29 joints；
- 16 bodies 均可在 live articulation 中按指定顺序解析；
- action space 是 23，wrist target 固定。

### M1：单 motion、RSI 和时序

实现：

- strict NPZ validator/loader；
- `G1MotionCommand`；
- exact RSI；
- `just_reset` temporal guard；
- motion-end truncation。

验收：

- synthetic 4-frame motion 中 reset 到 `k`，第一次 obs 目标严格是 `k+1`；
- action 后 reward 严格使用同一个 `k+1`；
- command promotion 后下一 obs 是 `k+2`；
- last frame 被奖励后才 reset；
- 无 hard-coded motion body indexes。

### M2：789-D oracle

实现 observation blocks、actor/critic group mapping和导出 layout metadata。

验收：

- 每个 block 的 shape、offset、数值测试通过；
- global translation/rotation invariance test 通过；
- 无 robot-state history、无 corruption；reference joint command 包含 `t, ..., t+4`；
- runner 打印 Actor=789、Critic=789、Action=23。

### M3：reward、termination、DR

验收：

- robot state 等于 target 时，各 tracking reward 接近 1；
- 单一关节/body perturbation 只影响预期 reward；
- active task 的 regularization/penalty 从 iteration 0 起就是完整权重；
- early failure 一次性 penalty 精确为 `-50`，timeout 为 0；
- gravity/link-distance threshold 边界测试通过；
- DR whitelist 通过。

### M4：Isaac Sim PPO smoke

阶梯执行：

1. 1 env reset/step 200 steps，无 NaN、无 wrist 漂移；
2. 16–64 env，2–10 PPO iterations；
3. 单 motion overfit，确认 tracking error 显著下降；
4. checkpoint play 可重复；
5. 4096 env 性能/显存测试；
6. 8192 env 生产配置。

smoke 阶段不引入多 motion、cache、multi-GPU，确保问题可以归因。

### M5：多 motion 生产训练

prepared-motion dataset loader 已支持：

- uniform motion sampling + uniform RSI；
- 记录 used motions；
- 单 NPZ、递归 directory 和相对路径 `.txt/.lst` manifest；
- clip-local termination，不跨轨迹取下一帧；
- 小集合 learnability；
- 最后扩到全量数据；当常驻内存成为瓶颈时再增加 memmap/cache。

failure-aware sampling、网络增大和 reward sigma 调整必须分别做实验，不能一次同时打开。

## 14. 关键测试清单

### 14.1 无 Isaac Sim 的纯 Python 测试

- 23/6/29 joint set 和顺序。
- 16-body list 唯一性。
- observation slice offsets 总和为 789。
- NPZ fields、names、shape、finite、quaternion norm、fps。
- 29-joint motion 到 23-joint controlled reorder。
- locked wrist reference tolerance。
- optional curriculum helper（active task 关闭）。
- local-frame math 与 rotation 6D。

### 14.2 需要 Isaac Sim 的 contract tests

- Gym registration 能解析 env/agent cfg。
- live articulation joint/body names。
- `merge_fixed_joints=False` 后 head/hand bodies 存在。
- action manager 23-D 且 preserve order。
- random action 后 wrist target/position 不漂移。
- contact sensor 的 feet body IDs 正确。
- RSI 写入 root、q、qdot 后数值一致。
- synthetic reference 的 `t -> t+1`。
- reward 和 termination 使用 target reference，不是 current reference。
- Train/Play observation dimensions和顺序完全一致。

## 15. 主要风险与处理

| 风险 | 后果 | 处理 |
| --- | --- | --- |
| 把外部标准 23-DoF URDF 当成“锁六 wrist” | policy 维度没错但物理语义全错 | 使用 29-DoF asset + 明确 23/6 lists；合同测试 |
| fixed links 被 URDF importer 合并 | 16-body observation 缺 head/hand | 首版 `merge_fixed_joints=False`；live body test |
| NPZ body order依赖 articulation order | reward/obs silently 错位 | 强制 `body_names` 并按名字重排 |
| reset 后 command manager 再 update | 第一次 goal 从 k+1 跳到 k+2 | `just_reset` mask + synthetic temporal test |
| reward 使用 ref[k]、obs 使用 ref[k+1] | PPO 学不到正确一步控制 | current/target 属性分离，reward target test |
| termination penalty 被 Isaac Lab 乘 dt | `-200` 实际变成 `-4` | indicator 除 `step_dt`，精确数值测试 |
| 未进入 action 的 wrist 成为 passive joint | 手部姿态漂移，body reward异常 | locked actuator group + drift test |
| 直接复制 BeyondMimic push/adaptive sampler | 偏离 Stage-1 oracle upper bound | DR whitelist，uniform RSI 默认 |
| 一开始全量 8192 env | 时序/shape bug难定位 | 1 env → 64 env → overfit → scale-up |
| G1 资产未进入 package data | editable install 能跑、部署安装找不到资产 | 更新 MANIFEST，安装后资源测试 |

## 16. 推荐实现顺序

按以下顺序提交，每一步都可独立验证：

1. `remove legacy teacher code/assets/docs and registry entries`。
2. `G1 asset package + 23/6/16 contracts`。
3. `G1 articulation + PD + 23-D action + wrist lock`。
4. `strict G1 motion schema + RSI + temporal alignment`。
5. `789-D oracle observation`。
6. `tracking rewards + early termination`。
7. `fixed regularization + asset-only DR`。
8. `Gym registration + PPO cfg + train/play scripts`。
9. `one-env and PPO smoke tests`。
10. `single-motion overfit report`。
11. `multi-motion loader/cache and production scale`。

首个真正的完成标准不是“脚本能启动”，而是：

```text
23-D action semantics correct
+ wrist genuinely locked
+ 16-body map correct
+ obs/reward both target the same ref[k+1]
+ one-motion tracking error learns downward
+ checkpoint can deterministic play
```

满足这些条件后，得到的 checkpoint 才能作为 Stage-2 DAgger 的 `a_oracle` 来源。

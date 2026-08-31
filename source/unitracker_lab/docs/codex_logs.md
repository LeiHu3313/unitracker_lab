# Codex change log

## 2026-08-31

- Teacher observation 改为 `teacher_state`（677D）+ `teacher_reference`（748D），actor / critic 输入均为 1425D。
- motion 数据 body state 改为 17 body；观测使用其中 14 body。新增 `rebuild_teacher_motion_body_states.py` 生成 `g1_lafan_40`。
- 修复 `teacher_reference` 的多帧 ground-anchor 广播。
- whole-body pose reward 改为 14 body；joint position / velocity reward 改为 25 DoF，去除左右 ankle pitch / roll。
- adaptive sampling 改为 14 body、reward sigma 归一化的 failure/error priority；失败时 80% 同 clip 回退 25–124 帧，末 50 帧和 timeout 保持全局自适应重采样。
- 保留 `data/g1_lafan_40`；新增其 `dataset.yaml`，删除旧的本地输入数据目录。

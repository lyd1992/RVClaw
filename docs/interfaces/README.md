# Interfaces

当前主线先保留最关键的联调接口：

## 已预留契约

- `contracts/task_state.schema.json`
- `contracts/device_status.schema.json`
- `contracts/inspection_result.schema.json`

## 后续应补的接口文档

1. `M2 -> M5` 任务状态字段定义
2. `M3 -> M4` 到点触发与截图接口
3. `M4 -> M5` 识别结果与告警结果接口
4. `M5 -> M2` 开始巡检 / 结束巡检 / 人工接管指令接口

## 约束

所有跨模块字段变更，先改契约，再改生产者和消费者。

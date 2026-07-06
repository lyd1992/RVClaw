# RVClaw

`RVClaw` 现阶段定位为 `RVClaw EdgeOne DevKit` 首发产品的软件主仓，面向 `K3 CoM260 Kit + V550 ROS2` 轮式底盘路线，服务于室内低速巡检 Demo 的开发、联调、演示和后续产品化。

## 当前阶段

当前主线不是旧的 `Demo Claw / K3 Pico-ITX / Agent Command Center` 方向，而是新的首发产品主线：

- 主控硬件：`K3 CoM260 Kit`
- 移动底座：`V550 ROS2`
- P0 核心模块：`M2` 底盘控制、`M3` 感知接入、`M4` 识别算法、`M5` Studio 前端
- 当前仓库状态：已完成主仓重布局与接口预留，后续按模块逐步落实现实装代码

## 历史保留

旧主线代码历史未丢失，已通过以下归档引用保留：

- archive tag: `archive/pre-edgeone-relayout-20260703`
- archive branch: `archive/pre-edgeone-relayout-20260703`

## 仓库结构

```text
contracts/                    # 模块间接口契约与 JSON schema 预留
docs/                         # 架构说明、接口文档、后续联调文档
robot/ros2_ws/src/            # ROS2 侧能力
  rvclaw_base/                # M2 底盘控制
  rvclaw_perception/          # M3 感知接入
  rvclaw_algo/                # M4 识别算法
  rvclaw_bringup/             # 整机 bring-up / launch / 集成编排
services/
  rvclaw_api/                 # 提供给 Studio 的本地接口层
  rvclaw_report/              # 报告生成与导出
studio/web/                   # M5 Studio 前端
configs/
  scenes/                     # 场景点位与演示路线配置
  thresholds/                 # 识别阈值、告警阈值
  device/                     # 设备与传感器配置
deploy/                       # 环境部署、启动脚本、设备侧脚本
scripts/                      # 项目辅助脚本
```

## 模块映射

| 模块 | 目录 | 职责 |
| --- | --- | --- |
| `M2` | `robot/ros2_ws/src/rvclaw_base/` | 底盘控制、任务状态、接管与停稳辅助 |
| `M3` | `robot/ros2_ws/src/rvclaw_perception/` | 相机、雷达、编码器、`IMU` 接入与到点触发 |
| `M4` | `robot/ros2_ws/src/rvclaw_algo/` | 巡检识别、异常判定、结构化结果输出 |
| `M5` | `studio/web/` + `services/rvclaw_api/` | 任务编排、前端展示、报告入口 |

## 预留接口

当前先预留 3 类核心契约：

- `contracts/task_state.schema.json`
- `contracts/device_status.schema.json`
- `contracts/inspection_result.schema.json`
- `contracts/person_tracking.schema.json`

后续所有 `M2 -> M3 -> M4 -> M5` 的联调，都优先围绕这些契约对齐。

## 开发原则

1. 先对齐契约，再补实现。
2. 优先打通首发演示闭环，不追求大而全平台。
3. 目录按产品域划分，不再回到旧的单体 demo app 结构。

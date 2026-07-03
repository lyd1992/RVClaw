# RVClaw EdgeOne Architecture

## 目标

围绕 `K3 CoM260 Kit + V550 ROS2` 的首发产品，建立一个以产品域为边界的主仓结构，让底盘控制、感知、识别、Studio 和报告输出能够并行开发、逐步联调。

## 分层方式

本仓库不再围绕旧的单体 demo app 展开，而是按产品域拆成 4 个核心执行面：

1. `robot/`：真实机器人侧运行能力
2. `services/`：本地服务层和报告生成
3. `studio/`：可视化界面与任务编排
4. `contracts/`：模块间稳定契约

## 模块关系

```text
M2 底盘控制
  -> 输出 task state / device status
M3 感知接入
  -> 输出 capture trigger / sensor status / snapshots
M4 识别算法
  -> 输出 inspection result / anomaly result
M5 Studio 与报告
  -> 消费 M2/M3/M4 输出，完成展示、控制与导出
```

## 当前设计原则

1. `main` 只保留新主线，不把旧 Demo Claw 代码搬回当前目录。
2. 契约先行，接口在 `contracts/` 中预留。
3. 真实实现逐步补入对应产品域，不混放到一个通用 `src/` 目录下。

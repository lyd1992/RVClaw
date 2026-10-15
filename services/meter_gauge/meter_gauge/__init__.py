"""RVClaw 指针仪表盘识别服务包。

注意：本 __init__ 刻意不 import 子模块 —— `algo` 需要 ncnn 绑定，
保持轻量以便在没有 ncnn 的环境里也能 import 包本身（例如跑纯逻辑测试）。
"""

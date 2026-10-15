#!/usr/bin/env python3
"""RVClaw 指针仪表盘识别服务 —— 启动入口。

    python3 run_server.py [--port 8018] [--host 0.0.0.0]

systemd 用绝对路径直接调本文件即可（包目录需在本文件同级）。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from meter_gauge.service import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

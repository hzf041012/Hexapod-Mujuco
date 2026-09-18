#!/usr/bin/env python3
"""
六足机器人 MuJoCo 仿真入口

直接运行即可启动交互式仿真：
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slope_viewer import run

if __name__ == "__main__":
    run()

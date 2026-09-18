# Hexapod-Mujoco

一个六足机器人，包含 MuJoCo 模型文件以及相关控制程序。

> 本仓库为毕业设计《仿生六足移动机器人设计》的代码部分。
> SolidWorks 模型、Ansys 仿真、论文与答辩材料等大体积二进制文件未纳入版本控制。

## 目录结构

```
.
├── MuJoCo/                  # 基于 MuJoCo 物理引擎的斜坡行走仿真
│   ├── algorithm/           # 算法层：步态生成、平衡控制、正逆运动学
│   ├── application/         # 应用层：分层架构主控制器
│   ├── hardware/            # 硬件接口层：MuJoCo 仿真环境封装
│   ├── utils/               # 键盘输入封装等工具
│   ├── meshes/              # 机器人 3D 网格（19 个 OBJ）
│   ├── data/                # 仿真数据输出目录
│   ├── test.xml             # MuJoCo 场景文件（含 15° 斜坡）
│   ├── stair_climbing_control.py   # 核心控制器（自包含完整实现）
│   ├── slope_viewer.py             # 斜坡交互式观察 + 数据记录
│   └── test_simulation.py          # 斜坡行走自动化测试
└── MATLAB/
    └── leg.m                # 腿部运动学计算
```

## 环境依赖

- Python >= 3.10
- MuJoCo >= 2.3
- numpy >= 1.24
- matplotlib >= 3.7

```bash
cd MuJoCo
pip install -r requirements.txt
```

## 快速开始

```bash
cd MuJoCo
python main.py                 # 项目入口
python slope_viewer.py         # 斜坡交互式观察
python test_simulation.py      # 斜坡行走自动化测试
```

MuJoCo 目录下的详细说明见 [MuJoCo/README.md](MuJoCo/README.md)。

## 作者

胡增凡

# 六足机器人斜坡行走 MuJoCo 仿真

基于 MuJoCo 物理引擎的六足机器人斜坡行走仿真项目。

## 项目结构

```
.
├── algorithm/                    # 算法层
│   ├── balance_controller.py     # 平衡控制
│   ├── gait_generator.py         # 步态生成
│   └── kinematics.py             # 运动学
├── application/                  # 应用层
│   └── hexapod_controller.py     # 分层架构主控制器
├── hardware/                     # 硬件接口层
│   └── robot_interface.py        # MuJoCo 仿真环境封装
├── utils/                        # 工具
│   └── keyboard_controller.py    # 键盘输入封装
├── meshes/                       # 机器人 3D 网格（20个 OBJ）
├── data/                         # 仿真数据存储目录
├── stair_climbing_control.py     # 核心控制器（自包含完整实现）
├── slope_viewer.py               # 斜坡交互式观察 + 数据记录
├── test_simulation.py            # 斜坡行走自动化测试
├── test.xml                      # MuJoCo 场景文件（含 15° 斜坡）
├── main.py                       # 项目入口
├── requirements.txt
└── README.md
```

## 环境依赖

- Python >= 3.10
- MuJoCo >= 2.3
- numpy
- matplotlib
- pandas + openpyxl（数据记录需要）

```bash
pip install -r requirements.txt
```

## 快速运行

### 1. 交互式仿真（推荐）

```bash
python main.py
```

键盘控制：
| 按键 | 功能 |
|------|------|
| W/S/A/D | 前进/后退/左移/右移 |
| Q/E | 原地左转/右转 |
| 1/2/3 | 切换 Tripod / Wave / Ripple 步态 |
| T | 切换平地/斜坡模式 |
| R | 开始/暂停数据记录 |
| Space | 紧急停止 |

> **斜坡模式**：按 `T` 切换，自动注入 15° 斜坡地形。退出后记录的数据自动保存到 `data/` 目录。

### 2. 斜坡行走自动化测试

```bash
python test_simulation.py
```

测试结果输出到 `results/` 目录：
- `results/slope_walking_results.png` — 轨迹与姿态对比图
- `results/slope_walking_report.md` — 详细测试报告

## 数据存储

交互模式下按 `R` 键记录的数据，退出后自动保存为：

```
data/slope_data_YYYYMMDD_HHMMSS.xlsx
```

包含字段：时间、躯干位置/姿态、指令速度、实际速度、各关节角度、足端接触数等。

## 核心功能

- **三种步态**：Tripod（最快）、Wave（最稳）、Ripple（平衡）
- **数值逆运动学**：基于 MuJoCo Jacobian，误差 < 1mm
- **地形自适应**：摆动相根据前方地形动态调整抬腿高度
- **双模式切换**：平地/斜坡一键切换（T键）

"""
键盘控制器模块
支持通过键盘实时控制六足机器人
使用方向键控制移动
"""

import numpy as np
from typing import Optional, Callable, Dict
from dataclasses import dataclass
from enum import Enum


class KeyState(Enum):
    """按键状态"""
    RELEASED = 0
    PRESSED = 1
    HOLD = 2


@dataclass
class ControlCommand:
    """控制指令"""
    vx: float = 0.0          # 前进速度 (m/s)
    vy: float = 0.0          # 侧向速度 (m/s)
    omega: float = 0.0       # 旋转速度 (rad/s)
    height_delta: float = 0.0  # 高度调整
    gait_change: bool = False  # 切换步态
    emergency_stop: bool = False  # 紧急停止
    stand_up: bool = False   # 站立
    sit_down: bool = False   # 坐下


class KeyboardController:
    """
    键盘控制器
    使用方向键控制移动，避免与MuJoCo默认快捷键冲突
    """
    
    def __init__(self):
        """初始化键盘控制器"""
        # 按键映射
        self.key_map = {
            'forward': ['up'],          # ↑ - 前进
            'backward': ['down'],       # ↓ - 后退
            'left': ['left'],           # ← - 左移
            'right': ['right'],         # → - 右移
            'turn_left': ['q'],         # Q - 左转
            'turn_right': ['e'],        # E - 右转
            'speed_up': ['m'],          # M - 加速
            'speed_down': ['n'],        # N - 减速
            'height_up': ['r'],         # R - 增加高度
            'height_down': ['f'],       # F - 降低高度
            'gait_switch': ['tab'],     # Tab - 切换步态
            'emergency_stop': ['space'],# Space - 紧急停止
            'stand': ['z'],             # Z - 站立
            'sit': ['x'],               # X - 坐下
            'reset': ['c'],             # C - 重置
            'quit': ['esc'],            # Esc - 退出
        }
        
        # 速度参数
        self.linear_speed = 0.08     # 基础线速度 (m/s)
        self.angular_speed = 0.4     # 基础角速度 (rad/s)
        self.height_speed = 0.01     # 高度调整速度 (m/s)
        self.speed_multiplier = 2.0  # 加速倍数
        
        # 当前状态
        self.current_speed_level = 1.0
        self.pressed_keys = set()
        
        # 回调函数
        self.on_gait_switch: Optional[Callable] = None
        self.on_emergency_stop: Optional[Callable] = None
        self.on_stand: Optional[Callable] = None
        self.on_sit: Optional[Callable] = None
        self.on_reset: Optional[Callable] = None
        self.on_quit: Optional[Callable] = None
        
        # 输入模式
        self.input_mode = "auto"
        self._keyboard_module = None
        
    def initialize(self, mode: str = "auto"):
        """
        初始化键盘输入
        
        Args:
            mode: 输入模式 ("keyboard", "auto")
        """
        self.input_mode = mode
        
        if mode in ["keyboard", "auto"]:
            try:
                import keyboard
                self._keyboard_module = keyboard
                print("键盘模块已加载")
            except ImportError:
                print("警告: 未安装keyboard模块")
                print("请运行: pip install keyboard")
                self._keyboard_module = None
    
    def update(self) -> ControlCommand:
        """
        更新键盘状态并返回控制指令
        
        Returns:
            ControlCommand: 控制指令
        """
        cmd = ControlCommand()
        
        if self.input_mode == "keyboard" and self._keyboard_module:
            cmd = self._update_keyboard_module()
        
        return cmd
    
    def _update_keyboard_module(self) -> ControlCommand:
        """使用keyboard模块更新"""
        cmd = ControlCommand()
        kb = self._keyboard_module
        
        if kb is None:
            return cmd
        
        # 检查退出 (Esc)
        if kb.is_pressed('esc'):
            cmd.quit = True
            if self.on_quit:
                self.on_quit()
            return cmd
        
        # 速度倍数
        speed_mult = self.speed_multiplier if kb.is_pressed('m') else 1.0
        if kb.is_pressed('n'):
            speed_mult *= 0.5
        
        # 线速度 (方向键)
        vx = 0.0
        vy = 0.0
        
        if kb.is_pressed('up'):      # 前进
            vx = self.linear_speed * speed_mult
        elif kb.is_pressed('down'):  # 后退
            vx = -self.linear_speed * speed_mult
        
        if kb.is_pressed('left'):    # 左移
            vy = self.linear_speed * speed_mult
        elif kb.is_pressed('right'): # 右移
            vy = -self.linear_speed * speed_mult
        
        cmd.vx = vx
        cmd.vy = vy
        
        # 角速度 (Q/E)
        if kb.is_pressed('q'):       # 左转
            cmd.omega = self.angular_speed * speed_mult
        elif kb.is_pressed('e'):     # 右转
            cmd.omega = -self.angular_speed * speed_mult
        
        # 高度调整 (R/F)
        if kb.is_pressed('r'):       # 增加高度
            cmd.height_delta = self.height_speed
        elif kb.is_pressed('f'):     # 降低高度
            cmd.height_delta = -self.height_speed
        
        # 切换步态 (Tab)
        if kb.is_pressed('tab'):
            cmd.gait_change = True
            if self.on_gait_switch:
                self.on_gait_switch()
        
        # 紧急停止 (Space)
        if kb.is_pressed('space'):
            cmd.emergency_stop = True
            if self.on_emergency_stop:
                self.on_emergency_stop()
        
        # 站立/坐下 (Z/X)
        if kb.is_pressed('z'):
            cmd.stand_up = True
            if self.on_stand:
                self.on_stand()
        
        if kb.is_pressed('x'):
            cmd.sit_down = True
            if self.on_sit:
                self.on_sit()
        
        # 重置 (C)
        if kb.is_pressed('c'):
            if self.on_reset:
                self.on_reset()
        
        return cmd
    
    def set_speed(self, linear: Optional[float] = None, 
                  angular: Optional[float] = None):
        """
        设置速度参数
        
        Args:
            linear: 线速度
            angular: 角速度
        """
        if linear is not None:
            self.linear_speed = linear
        if angular is not None:
            self.angular_speed = angular
    
    def get_help_text(self) -> str:
        """获取帮助文本"""
        return """
╔══════════════════════════════════════════════════════════════════╗
║                    六足机器人键盘控制说明                          ║
╠══════════════════════════════════════════════════════════════════╣
║  移动控制:                                                        ║
║    ↑ / ↓       - 前进 / 后退                                     ║
║    ← / →       - 左移 / 右移                                     ║
║    Q / E       - 左转 / 右转                                     ║
║    M           - 加速 (2倍)                                      ║
║    N           - 减速 (0.5倍)                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  姿态控制:                                                        ║
║    R           - 增加身体高度                                    ║
║    F           - 降低身体高度                                    ║
╠══════════════════════════════════════════════════════════════════╣
║  功能按键:                                                        ║
║    Tab         - 切换步态 (三角 → 波浪 → 涟漪)                  ║
║    Space       - 紧急停止                                        ║
║    Z           - 站立                                            ║
║    X           - 坐下                                            ║
║    C           - 重置位置                                        ║
║    Esc         - 退出程序                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  MuJoCo默认快捷键 (仍可用):                                      ║
║    0-9         - 切换相机视图                                    ║
║    Tab         - 切换相机模式 (在MuJoCo中)                       ║
║    F1-F12      - 各种显示选项                                    ║
║    鼠标        - 旋转/平移/缩放视角                              ║
╚══════════════════════════════════════════════════════════════════╝
"""


class InteractiveController:
    """
    交互式控制器
    整合键盘控制和可视化
    """
    
    def __init__(self, hexapod_controller, visualizer):
        """
        初始化交互式控制器
        
        Args:
            hexapod_controller: 六足机器人控制器
            visualizer: 可视化器
        """
        self.controller = hexapod_controller
        self.visualizer = visualizer
        self.keyboard = KeyboardController()
        
        # 状态
        self.is_running = False
        self.is_paused = False
        self.current_gait_idx = 0
        self.gait_types = ['tripod', 'wave', 'ripple']
        
        # 显示信息
        self.show_help = True
        self.show_info = True
        self.info_update_interval = 0.5
        self.last_info_update = 0.0
        
        # 设置回调
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """设置键盘回调"""
        self.keyboard.on_gait_switch = self._switch_gait
        self.keyboard.on_emergency_stop = self._emergency_stop
        self.keyboard.on_stand = self._stand_up
        self.keyboard.on_sit = self._sit_down
        self.keyboard.on_reset = self._reset
        self.keyboard.on_quit = self._quit
    
    def _switch_gait(self):
        """切换步态"""
        self.current_gait_idx = (self.current_gait_idx + 1) % len(self.gait_types)
        new_gait = self.gait_types[self.current_gait_idx]
        
        # 更新控制器步态
        from algorithm.gait_generator import TripodGait, WaveGait, RippleGait
        gait_classes = {
            'tripod': TripodGait,
            'wave': WaveGait,
            'ripple': RippleGait
        }
        params = self.controller.gait.params
        self.controller.gait = gait_classes[new_gait](params)
        
        print(f"\n[步态切换] 当前步态: {new_gait.upper()}")
    
    def _emergency_stop(self):
        """紧急停止"""
        print("\n[紧急停止]")
        self.controller.set_velocity_command(0, 0, 0)
        self.controller.emergency_stop()
    
    def _stand_up(self):
        """站立"""
        print("\n[站立]")
        self.controller.stand_up(duration=1.5)
    
    def _sit_down(self):
        """坐下"""
        print("\n[坐下]")
        self.controller.sit_down(duration=1.5)
    
    def _reset(self):
        """重置"""
        print("\n[重置位置]")
        self.controller.reset()
        self.controller.stand_up(duration=1.0)
    
    def _quit(self):
        """退出"""
        print("\n[退出程序]")
        self.is_running = False
    
    def run(self, duration: Optional[float] = None):
        """
        运行交互式控制
        
        Args:
            duration: 运行时长（秒），None表示无限运行
        """
        import time
        
        # 初始化键盘
        self.keyboard.initialize()
        
        # 打印帮助
        print(self.keyboard.get_help_text())
        
        # 重置并站立
        print("初始化机器人...")
        self.controller.reset()
        self.controller.stand_up(duration=2.0)
        
        # 启动主循环
        print("\n控制循环已启动，使用键盘控制机器人")
        print("=" * 50)
        
        self.is_running = True
        start_time = time.time()
        dt = 1.0 / self.controller.params.control_freq
        
        # 速度平滑
        target_vx, target_vy, target_omega = 0.0, 0.0, 0.0
        current_vx, current_vy, current_omega = 0.0, 0.0, 0.0
        smoothing_factor = 0.1
        
        try:
            while self.is_running:
                # 检查运行时长
                if duration and (time.time() - start_time) > duration:
                    break
                
                # 获取键盘指令
                cmd = self.keyboard.update()
                
                # 处理特殊命令
                if cmd.quit:
                    break
                
                if cmd.stand_up:
                    self.controller.stand_up(duration=1.5)
                    continue
                
                if cmd.sit_down:
                    self.controller.sit_down(duration=1.5)
                    continue
                
                # 更新目标速度
                if not cmd.emergency_stop:
                    target_vx = cmd.vx
                    target_vy = cmd.vy
                    target_omega = cmd.omega
                else:
                    target_vx = target_vy = target_omega = 0.0
                
                # 速度平滑
                current_vx += (target_vx - current_vx) * smoothing_factor
                current_vy += (target_vy - current_vy) * smoothing_factor
                current_omega += (target_omega - current_omega) * smoothing_factor
                
                # 设置速度指令
                self.controller.set_velocity_command(current_vx, current_vy, current_omega)
                
                # 更新控制器
                state = self.controller.update()
                
                # 更新可视化
                if self.visualizer:
                    self.visualizer.update_viewer()
                    
                    # 显示状态信息
                    if self.show_info and (time.time() - self.last_info_update) > self.info_update_interval:
                        self._display_status(state)
                        self.last_info_update = time.time()
                
                # 控制频率
                time.sleep(dt)
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            # 清理
            print("\n清理中...")
            self.controller.set_velocity_command(0, 0, 0)
            self.controller.sit_down(duration=1.5)
            print("程序已退出")
    
    def _display_status(self, state):
        """显示状态信息"""
        full_state = self.controller.get_state()
        
        # 清屏并显示状态
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("           六足机器人实时状态")
        print("=" * 60)
        
        # 位置信息
        pos = full_state['torso_position']
        print(f"位置: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
        
        # 姿态信息
        roll = np.rad2deg(full_state['roll'])
        pitch = np.rad2deg(full_state['pitch'])
        yaw = np.rad2deg(full_state['yaw'])
        print(f"姿态: Roll={roll:6.2f}\u00b0, Pitch={pitch:6.2f}\u00b0, Yaw={yaw:6.2f}\u00b0")
        
        # 速度指令
        vel = self.controller.current_velocity
        print(f"速度: Vx={vel[0]:.3f}m/s, Vy={vel[1]:.3f}m/s, \u03c9={vel[2]:.3f}rad/s")
        
        # 步态信息
        gait_name = self.gait_types[self.current_gait_idx].upper()
        print(f"步态: {gait_name}")
        
        # 足底接触
        contacts = full_state['foot_contacts']
        contact_str = ''.join(['●' if c else '○' for c in contacts])
        print(f"接触: {contact_str} (RF RM RR LF LM LR)")
        
        # 仿真时间
        print(f"时间: {full_state['time']:.2f}s")
        
        print("=" * 60)
        print("按 H 显示帮助信息")
        print("=" * 60)

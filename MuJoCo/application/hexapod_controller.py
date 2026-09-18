"""
六足机器人主控制器
整合硬件层和算法层，提供统一的控制接口
"""

import numpy as np
import mujoco
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import time
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入硬件层
from hardware.robot_interface import RobotInterface, RobotState
from hardware.sensor_interface import SensorInterface, IMUData
from hardware.actuator_interface import ActuatorInterface

# 导入算法层
from algorithm.kinematics import HexapodKinematics, LegConfiguration
from algorithm.gait_generator import GaitParameters, TripodGait, WaveGait, RippleGait
from algorithm.balance_controller import BalanceController, BalanceParameters
from algorithm.trajectory_planner import TrajectoryPlanner


@dataclass
class ControlParameters:
    """控制参数"""
    # 步态参数
    gait_type: str = "tripod"  # tripod, wave, ripple
    step_height: float = 0.03
    step_length: float = 0.08
    cycle_time: float = 1.0
    
    # 平衡参数
    balance_enabled: bool = True
    target_height: float = 0.12
    
    # 控制频率
    control_freq: float = 100.0  # Hz


class HexapodController:
    """
    六足机器人主控制器
    整合所有硬件和算法模块，提供高层控制接口
    """
    
    def __init__(self, model_path: str, params: Optional[ControlParameters] = None):
        """
        初始化控制器
        
        Args:
            model_path: MuJoCo模型文件路径
            params: 控制参数
        """
        self.params = params or ControlParameters()
        
        # 初始化硬件层
        self.robot = RobotInterface(model_path)
        self.sensors = SensorInterface(self.robot)
        self.actuators = ActuatorInterface(self.robot)
        
        # 初始化算法层
        self.kinematics = HexapodKinematics(LegConfiguration())
        self.trajectory_planner = TrajectoryPlanner()
        
        # 初始化步态生成器
        self._init_gait_generator()
        
        # 初始化平衡控制器
        balance_params = BalanceParameters(target_height=self.params.target_height)
        self.balance_controller = BalanceController(balance_params)
        
        # 控制状态
        self.is_running = False
        self.current_velocity = np.zeros(3)  # [vx, vy, omega]
        self.current_time = 0.0
        
        # 足端home位置（相对于身体）
        self.foot_home_positions = self._init_foot_home_positions()
        
        # 控制周期
        self.dt = 1.0 / self.params.control_freq
        
    def _init_gait_generator(self):
        """初始化步态生成器"""
        gait_params = GaitParameters(
            step_height=self.params.step_height,
            step_length=self.params.step_length,
            cycle_time=self.params.cycle_time
        )
        
        if self.params.gait_type == "tripod":
            self.gait = TripodGait(gait_params)
        elif self.params.gait_type == "wave":
            self.gait = WaveGait(gait_params)
        elif self.params.gait_type == "ripple":
            self.gait = RippleGait(gait_params)
        else:
            self.gait = TripodGait(gait_params)
    
    def _init_foot_home_positions(self) -> np.ndarray:
        """初始化足端home位置"""
        # 相对于躯干的默认站立位置
        home_positions = np.array([
            [-0.10, -0.08, -0.08],   # RF
            [-0.12,  0.00, -0.08],   # RM
            [-0.10,  0.08, -0.08],   # RR
            [ 0.10,  0.08, -0.08],   # LF
            [ 0.12,  0.00, -0.08],   # LM
            [ 0.10, -0.08, -0.08],   # LR
        ])
        return home_positions
    
    def reset(self, torso_pos: Optional[np.ndarray] = None,
              torso_quat: Optional[np.ndarray] = None):
        """
        重置机器人
        
        Args:
            torso_pos: 躯干初始位置
            torso_quat: 躯干初始姿态
        """
        # 默认初始位置和姿态
        if torso_pos is None:
            torso_pos = np.array([0.1, 0, 0.12])  # 匹配XML初始高度
        if torso_quat is None:
            torso_quat = np.array([1, 0, 0, 0])  # 单位四元数
        
        # 设置初始关节角度（站立姿态）
        initial_joint_pos = self._compute_stance_joint_angles()
        
        self.robot.reset(torso_pos, torso_quat, initial_joint_pos)
        self.balance_controller.reset()
        self.current_time = 0.0
        self.current_velocity = np.zeros(3)
        
    def _get_leg_joint_ids(self, leg_idx: int) -> list:
        """获取指定腿的关节ID列表"""
        model = self.robot.model
        leg_names = ['RF', 'RM', 'RR', 'LF', 'LM', 'LR']
        leg_name = leg_names[leg_idx]
        
        joint_ids = []
        for j in range(3):
            # Try different naming conventions
            jnt_name = f"leg{leg_idx+1}_revolute{j+1}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid < 0:
                jnt_name = f"Joint_{leg_idx+1}{j+1}"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                joint_ids.append(jid)
        return joint_ids
    
    def _compute_stance_joint_angles(self) -> np.ndarray:
        """计算站立姿态的关节角度（使用数值IK确保与MuJoCo一致）"""
        joint_angles = np.zeros(18)
        leg_names = ['RF', 'RM', 'RR', 'LF', 'LM', 'LR']
        
        # 保存当前状态
        original_qpos = self.robot.data.qpos.copy()
        original_qvel = self.robot.data.qvel.copy()
        
        for i in range(6):
            leg_name = leg_names[i]
            
            # 使用正确高度的home位置
            home_pos = self.foot_home_positions[i].copy()
            home_pos[2] = -self.params.target_height
            
            # 使用数值逆运动学
            leg_joint_ids = self._get_leg_joint_ids(i)
            leg_angles = self.kinematics.inverse_kinematics_leg_numerical(
                leg_name, home_pos,
                self.robot.model, self.robot.data,
                leg_joint_ids
            )
            joint_angles[i*3:(i+1)*3] = leg_angles
        
        # 恢复状态
        self.robot.data.qpos[:] = original_qpos
        self.robot.data.qvel[:] = original_qvel
        mujoco.mj_forward(self.robot.model, self.robot.data)
        
        return joint_angles
    
    def set_velocity_command(self, vx: float, vy: float = 0.0, omega: float = 0.0):
        """
        设置速度指令
        
        Args:
            vx: 前进速度 (m/s)
            vy: 侧向速度 (m/s)
            omega: 旋转速度 (rad/s)
        """
        self.current_velocity = np.array([vx, vy, omega])
    
    def update(self) -> RobotState:
        """
        更新控制器（主控制循环）
        
        Returns:
            RobotState: 当前机器人状态
        """
        # 获取传感器数据
        imu_data = self.sensors.get_imu_data()
        foot_contacts = self.sensors.get_foot_contact_states()
        foot_positions_world = self.robot.get_foot_positions()
        
        # 获取欧拉角
        roll, pitch, yaw = imu_data.get_euler_angles()
        
        # 获取当前高度估计
        height = self.sensors.get_height_estimate()
        
        # 计算平衡调整
        foot_adjustments = np.zeros((6, 3))
        if self.params.balance_enabled:
            foot_adjustments = self.balance_controller.compute_foot_adjustments(
                roll, pitch, height, foot_positions_world, foot_contacts, self.dt
            )
        
        # 计算目标关节角度
        target_joint_angles = self._compute_target_joint_angles(
            imu_data, foot_contacts, foot_adjustments
        )
        
        # 发送控制命令
        self.actuators.set_joint_positions(target_joint_angles, smooth=True)
        
        # 步态更新
        self.gait.update(self.dt, self.current_velocity)
        self.current_time += self.dt
        
        # 执行仿真步
        state = self.robot.step()
        
        return state
    
    def _compute_target_joint_angles(self, imu_data: IMUData,
                                      foot_contacts: np.ndarray,
                                      foot_adjustments: np.ndarray) -> np.ndarray:
        """
        计算目标关节角度（使用数值IK确保与MuJoCo一致）
        
        Args:
            imu_data: IMU数据
            foot_contacts: 足底接触状态
            foot_adjustments: 足端位置调整量
            
        Returns:
            np.ndarray: 目标关节角度 [18]
        """
        target_angles = np.zeros(18)
        leg_names = ['RF', 'RM', 'RR', 'LF', 'LM', 'LR']
        
        # 保存当前状态
        original_qpos = self.robot.data.qpos.copy()
        
        for i, leg_name in enumerate(leg_names):
            # 获取足端轨迹
            home_pos = self.foot_home_positions[i].copy()
            
            # 同步身体高度
            home_pos[2] = -self.params.target_height
            
            # 应用平衡调整
            home_pos += foot_adjustments[i]
            
            # 获取步态轨迹
            target_pos, _ = self.gait.get_foot_trajectory(
                i, home_pos, self.current_velocity
            )
            
            # 使用数值逆运动学
            leg_joint_ids = self._get_leg_joint_ids(i)
            leg_angles = self.kinematics.inverse_kinematics_leg_numerical(
                leg_name, target_pos,
                self.robot.model, self.robot.data,
                leg_joint_ids
            )
            
            target_angles[i*3:(i+1)*3] = leg_angles
        
        # 恢复状态
        self.robot.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.robot.model, self.robot.data)
        
        return target_angles
    
    def stand_up(self, duration: float = 2.0):
        """
        站立动作（使用数值IK确保与MuJoCo一致）
        
        Args:
            duration: 站立过程持续时间
        """
        stand_height = self.params.target_height
        
        num_steps = int(duration / self.dt)
        
        # 保存当前状态
        original_qpos = self.robot.data.qpos.copy()
        
        for step in range(num_steps):
            target_angles = np.zeros(18)
            leg_names = ['RF', 'RM', 'RR', 'LF', 'LM', 'LR']
            
            for i, leg_name in enumerate(leg_names):
                home_pos = self.foot_home_positions[i].copy()
                home_pos[2] = -stand_height
                
                # 使用数值逆运动学
                leg_joint_ids = self._get_leg_joint_ids(i)
                leg_angles = self.kinematics.inverse_kinematics_leg_numerical(
                    leg_name, home_pos,
                    self.robot.model, self.robot.data,
                    leg_joint_ids
                )
                target_angles[i*3:(i+1)*3] = leg_angles
            
            # 恢复qpos的其他部分（躯干位置和姿态）
            current_torso = self.robot.data.qpos[:7].copy()
            self.robot.data.qpos[:] = original_qpos
            self.robot.data.qpos[:7] = current_torso
            
            self.actuators.set_joint_positions(target_angles, smooth=False)
            self.robot.step()
    
    def sit_down(self, duration: float = 2.0):
        """
        坐下动作
        
        Args:
            duration: 坐下过程持续时间
        """
        crouch_height = 0.05
        stand_height = self.params.target_height
        
        num_steps = int(duration / self.dt)
        
        for step in range(num_steps):
            progress = step / num_steps
            
            # 插值高度
            current_height = stand_height + (crouch_height - stand_height) * progress
            
            # 调整足端位置
            target_angles = np.zeros(18)
            leg_names = ['RF', 'RM', 'RR', 'LF', 'LM', 'LR']
            
            for i, leg_name in enumerate(leg_names):
                home_pos = self.foot_home_positions[i].copy()
                home_pos[2] = -current_height
                
                target_pos_leg = self.kinematics.body_to_leg_frame(leg_name, home_pos)
                leg_angles = self.kinematics.inverse_kinematics_leg(leg_name, target_pos_leg)
                target_angles[i*3:(i+1)*3] = leg_angles
            
            self.actuators.set_joint_positions(target_angles, smooth=False)
            self.robot.step()
    
    def get_state(self) -> Dict:
        """
        获取完整状态信息
        
        Returns:
            Dict: 状态字典
        """
        robot_state = self.robot.get_state()
        imu_data = self.sensors.get_imu_data()
        roll, pitch, yaw = imu_data.get_euler_angles()
        
        return {
            'time': robot_state.time,
            'torso_position': robot_state.torso_position,
            'torso_orientation': robot_state.torso_orientation,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'joint_positions': robot_state.joint_positions,
            'joint_velocities': robot_state.joint_velocities,
            'foot_contacts': robot_state.foot_contacts,
            'imu_acceleration': robot_state.imu_acceleration,
            'imu_gyroscope': robot_state.imu_gyroscope,
        }
    
    def emergency_stop(self):
        """紧急停止"""
        self.current_velocity = np.zeros(3)
        self.actuators.emergency_stop()

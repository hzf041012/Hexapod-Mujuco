"""
机器人硬件接口 - 封装MuJoCo仿真环境
"""

import mujoco
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class RobotState:
    """机器人状态数据结构"""
    # 关节状态
    joint_positions: np.ndarray      # 关节角度 [18]
    joint_velocities: np.ndarray     # 关节速度 [18]
    joint_torques: np.ndarray        # 关节力矩 [18]
    
    # 躯干状态
    torso_position: np.ndarray       # 躯干位置 [3]
    torso_orientation: np.ndarray    # 躯干姿态（四元数）[4]
    torso_velocity: np.ndarray       # 躯干线速度 [3]
    torso_angular_vel: np.ndarray    # 躯干角速度 [3]
    
    # IMU数据
    imu_acceleration: np.ndarray     # IMU加速度 [3]
    imu_gyroscope: np.ndarray        # IMU陀螺仪 [3]
    imu_orientation: np.ndarray      # IMU姿态四元数 [4]
    
    # 接触状态
    foot_contacts: np.ndarray        # 足底接触状态 [6]
    
    # 时间
    time: float


class RobotInterface:
    """
    六足机器人硬件接口
    封装MuJoCo的底层操作，提供高层硬件访问接口
    """
    
    # 关节名称映射
    JOINT_NAMES = [
        'Joint_11', 'Joint_12', 'Joint_13',  # 腿1 - 右前
        'Joint_21', 'Joint_22', 'Joint_23',  # 腿2 - 右中
        'Joint_31', 'Joint_32', 'Joint_33',  # 腿3 - 右后
        'Joint_41', 'Joint_42', 'Joint_43',  # 腿4 - 左前
        'Joint_51', 'Joint_52', 'Joint_53',  # 腿5 - 左中
        'Joint_61', 'Joint_62', 'Joint_63',  # 腿6 - 左后
    ]
    
    # 执行器名称映射
    ACTUATOR_NAMES = [
        'Joint_11_servo', 'Joint_12_servo', 'Joint_13_servo',
        'Joint_21_servo', 'Joint_22_servo', 'Joint_23_servo',
        'Joint_31_servo', 'Joint_32_servo', 'Joint_33_servo',
        'Joint_41_servo', 'Joint_42_servo', 'Joint_43_servo',
        'Joint_51_servo', 'Joint_52_servo', 'Joint_53_servo',
        'Joint_61_servo', 'Joint_62_servo', 'Joint_63_servo',
    ]
    
    # 足底几何体名称
    FOOT_GEOM_NAMES = ['foot1_geom', 'foot2_geom', 'foot3_geom', 
                       'foot4_geom', 'foot5_geom', 'foot6_geom']
    
    # 腿分组
    LEG_INDICES = {
        'RF': [0, 1, 2],   # 右前 Right Front
        'RM': [3, 4, 5],   # 右中 Right Middle
        'RR': [6, 7, 8],   # 右后 Right Rear
        'LF': [9, 10, 11], # 左前 Left Front
        'LM': [12, 13, 14],# 左中 Left Middle
        'LR': [15, 16, 17] # 左后 Left Rear
    }
    
    def __init__(self, model_path: str):
        """
        初始化机器人接口
        
        Args:
            model_path: MuJoCo XML模型文件路径
        """
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 获取关节ID映射
        self.joint_ids = self._get_joint_ids()
        self.actuator_ids = self._get_actuator_ids()
        self.foot_geom_ids = self._get_foot_geom_ids()
        
        # 获取IMU传感器ID
        self.imu_orientation_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_orientation')
        self.imu_accel_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_accelerometer')
        self.imu_gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
        
        # 获取躯干body ID
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        
        # 获取自由关节ID
        self.free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'base_freejoint')
        
        # 仿真参数
        self.timestep = self.model.opt.timestep
        self.num_joints = len(self.JOINT_NAMES)
        self.num_legs = 6
        
    def _get_joint_ids(self) -> Dict[str, int]:
        """获取关节ID映射"""
        joint_ids = {}
        for name in self.JOINT_NAMES:
            joint_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return joint_ids
    
    def _get_actuator_ids(self) -> Dict[str, int]:
        """获取执行器ID映射"""
        actuator_ids = {}
        for name in self.ACTUATOR_NAMES:
            actuator_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return actuator_ids
    
    def _get_foot_geom_ids(self) -> List[int]:
        """获取足底几何体ID"""
        foot_ids = []
        for name in self.FOOT_GEOM_NAMES:
            foot_ids.append(mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
        return foot_ids
    
    def reset(self, torso_pos: Optional[np.ndarray] = None,
              torso_quat: Optional[np.ndarray] = None,
              joint_pos: Optional[np.ndarray] = None):
        """
        重置机器人状态
        
        Args:
            torso_pos: 躯干初始位置 [3]
            torso_quat: 躯干初始姿态（四元数）[4]
            joint_pos: 关节初始角度 [18]
        """
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置躯干位置和姿态
        if torso_pos is not None:
            self.data.qpos[self.free_joint_id:self.free_joint_id+3] = torso_pos
        if torso_quat is not None:
            self.data.qpos[self.free_joint_id+3:self.free_joint_id+7] = torso_quat
            
        # 设置关节角度
        if joint_pos is not None:
            for i, name in enumerate(self.JOINT_NAMES):
                joint_id = self.joint_ids[name]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_adr] = joint_pos[i]
        
        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)
    
    def set_joint_positions(self, positions: np.ndarray):
        """
        设置目标关节位置（位置控制）
        
        Args:
            positions: 目标关节角度 [18]
        """
        for i, name in enumerate(self.ACTUATOR_NAMES):
            actuator_id = self.actuator_ids[name]
            self.data.ctrl[actuator_id] = positions[i]
    
    def set_joint_torques(self, torques: np.ndarray):
        """
        设置关节力矩（力矩控制模式）
        
        Args:
            torques: 目标关节力矩 [18]
        """
        for i, name in enumerate(self.ACTUATOR_NAMES):
            actuator_id = self.actuator_ids[name]
            # 临时切换到力矩控制
            self.model.actuator_gainprm[actuator_id, 0] = 0
            self.data.ctrl[actuator_id] = torques[i]
    
    def step(self) -> RobotState:
        """
        执行一步仿真
        
        Returns:
            RobotState: 当前机器人状态
        """
        mujoco.mj_step(self.model, self.data)
        return self.get_state()
    
    def get_state(self) -> RobotState:
        """
        获取当前机器人状态
        
        Returns:
            RobotState: 机器人状态
        """
        # 获取关节状态
        joint_positions = np.zeros(self.num_joints)
        joint_velocities = np.zeros(self.num_joints)
        
        for i, name in enumerate(self.JOINT_NAMES):
            joint_id = self.joint_ids[name]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]
            joint_positions[i] = self.data.qpos[qpos_adr]
            joint_velocities[i] = self.data.qvel[qvel_adr]
        
        # 获取关节力矩
        joint_torques = self.data.actuator_force[:self.num_joints]
        
        # 获取躯干状态
        torso_position = self.data.qpos[self.free_joint_id:self.free_joint_id+3].copy()
        torso_orientation = self.data.qpos[self.free_joint_id+3:self.free_joint_id+7].copy()
        torso_velocity = self.data.qvel[self.free_joint_id:self.free_joint_id+3].copy()
        torso_angular_vel = self.data.qvel[self.free_joint_id+3:self.free_joint_id+6].copy()
        
        # 获取IMU数据
        sensor_adr = self.model.sensor_adr
        imu_orientation = self.data.sensordata[sensor_adr[self.imu_orientation_id]:
                                                sensor_adr[self.imu_orientation_id]+4].copy()
        imu_acceleration = self.data.sensordata[sensor_adr[self.imu_accel_id]:
                                                 sensor_adr[self.imu_accel_id]+3].copy()
        imu_gyroscope = self.data.sensordata[sensor_adr[self.imu_gyro_id]:
                                              sensor_adr[self.imu_gyro_id]+3].copy()
        
        # 获取足底接触状态
        foot_contacts = self._get_foot_contacts()
        
        return RobotState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            torso_position=torso_position,
            torso_orientation=torso_orientation,
            torso_velocity=torso_velocity,
            torso_angular_vel=torso_angular_vel,
            imu_acceleration=imu_acceleration,
            imu_gyroscope=imu_gyroscope,
            imu_orientation=imu_orientation,
            foot_contacts=foot_contacts,
            time=self.data.time
        )
    
    def _get_foot_contacts(self) -> np.ndarray:
        """
        获取足底接触状态
        
        Returns:
            np.ndarray: 接触状态 [6]，1表示接触，0表示未接触
        """
        contacts = np.zeros(self.num_legs)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for leg_idx, geom_id in enumerate(self.foot_geom_ids):
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    contacts[leg_idx] = 1
        return contacts
    
    def get_foot_positions(self) -> np.ndarray:
        """
        获取足底在世界坐标系中的位置
        
        Returns:
            np.ndarray: 足底位置 [6, 3]
        """
        foot_positions = np.zeros((self.num_legs, 3))
        for i, geom_name in enumerate(self.FOOT_GEOM_NAMES):
            geom_id = self.foot_geom_ids[i]
            foot_positions[i] = self.data.geom_xpos[geom_id].copy()
        return foot_positions
    
    def get_body_position(self, body_name: str) -> np.ndarray:
        """
        获取指定body在世界坐标系中的位置
        
        Args:
            body_name: body名称
            
        Returns:
            np.ndarray: 位置 [3]
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取关节限制
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (下限, 上限) 每个[18]
        """
        lower_limits = np.zeros(self.num_joints)
        upper_limits = np.zeros(self.num_joints)
        
        for i, name in enumerate(self.JOINT_NAMES):
            joint_id = self.joint_ids[name]
            lower_limits[i] = self.model.jnt_range[joint_id, 0]
            upper_limits[i] = self.model.jnt_range[joint_id, 1]
            
        return lower_limits, upper_limits
    
    def get_leg_joint_positions(self, leg_name: str) -> np.ndarray:
        """
        获取指定腿的关节角度
        
        Args:
            leg_name: 腿名称 ('RF', 'RM', 'RR', 'LF', 'LM', 'LR')
            
        Returns:
            np.ndarray: 关节角度 [3]
        """
        indices = self.LEG_INDICES[leg_name]
        state = self.get_state()
        return state.joint_positions[indices]

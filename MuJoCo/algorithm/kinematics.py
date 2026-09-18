"""
六足机器人运动学模块
包含正运动学和逆运动学计算
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LegConfiguration:
    """腿部几何参数"""
    # FIX: Updated link lengths to match actual MuJoCo model geometry
    # Old values: coxa=0.035, femur=0.065, tibia=0.085 (did not match XML model)
    # New values measured from MuJoCo body positions at zero joint angles
    coxa_length: float = 0.0584    # 根部连杆长度 (m) - measured from j1 to j2
    femur_length: float = 0.0655   # 大腿长度 (m) - measured from j2 to j3
    tibia_length: float = 0.1132  # 小腿长度 (m) - measured from j3 to foot tip
    
    # FIX: Updated mount positions with correct Z offset from XML model
    # XML model has Z offset of approximately -0.0042m for all leg mounts
    mount_positions = {
        'RF': np.array([-0.062, -0.099, -0.0042]),   # 右前
        'RM': np.array([-0.105, -0.002, -0.0042]),   # 右中
        'RR': np.array([-0.065, 0.097, -0.0042]),    # 右后
        'LF': np.array([0.062, 0.099, -0.0042]),     # 左前
        'LM': np.array([0.105, 0.002, -0.0042]),     # 左中
        'LR': np.array([0.065, -0.097, -0.0042]),    # 左后
    }
    
    # 腿部安装角度（相对于躯干）
    mount_angles = {
        'RF': np.deg2rad(180),   # 右前 - 与XML euler="0 0 180"一致
        'RM': np.deg2rad(180),   # 右中
        'RR': np.deg2rad(180),   # 右后
        'LF': np.deg2rad(180),   # 左前
        'LM': np.deg2rad(180),   # 左中
        'LR': np.deg2rad(180),   # 左后
    }
    
    # 关节方向符号 (处理左右腿关节2、3运动方向相反的问题)
    # 关节1(coxa)所有腿方向相同, 关节2(femur)和关节3(tibia)左右腿相反
    # 右腿: [1, 1, 1], 左腿: [1, -1, -1]
    joint_signs = {
        'RF': np.array([1, 1, 1]),
        'RM': np.array([1, 1, 1]),
        'RR': np.array([1, 1, 1]),
        'LF': np.array([1, -1, -1]),
        'LM': np.array([1, -1, -1]),
        'LR': np.array([1, -1, -1]),
    }


class HexapodKinematics:
    """
    六足机器人运动学
    提供单腿和全身运动学计算
    """
    
    def __init__(self, config: Optional[LegConfiguration] = None):
        """
        初始化运动学
        
        Args:
            config: 腿部配置参数
        """
        self.config = config or LegConfiguration()
        
        # 缓存三角函数值
        self._init_leg_transforms()
    
    def _init_leg_transforms(self):
        """初始化腿部变换矩阵"""
        self.leg_transforms = {}
        for leg_name in ['RF', 'RM', 'RR', 'LF', 'LM', 'LR']:
            angle = self.config.mount_angles[leg_name]
            self.leg_transforms[leg_name] = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
    
    def forward_kinematics_leg(self, leg_name: str, 
                                joint_angles: np.ndarray) -> np.ndarray:
        """
        单腿正运动学
        
        Args:
            leg_name: 腿名称 ('RF', 'RM', 'RR', 'LF', 'LM', 'LR')
            joint_angles: 关节角度 [3] (coxa, femur, tibia) in radians
            
        Returns:
            np.ndarray: 足底在腿坐标系中的位置 [3]
        """
        # 应用关节方向符号（内部计算使用带符号的角度）
        signs = self.config.joint_signs[leg_name]
        theta1 = joint_angles[0] * signs[0]
        theta2 = joint_angles[1] * signs[1]
        theta3 = joint_angles[2] * signs[2]
        
        L1 = self.config.coxa_length
        L2 = self.config.femur_length
        L3 = self.config.tibia_length
        
        # 计算足底位置（腿局部坐标系）
        # FIX: Z方向符号修正，确保足端在Z负方向（腿坐标系下方）
        x = L1 * np.cos(theta1) + L2 * np.cos(theta1) * np.cos(theta2) + \
            L3 * np.cos(theta1) * np.cos(theta2 + theta3)
        y = L1 * np.sin(theta1) + L2 * np.sin(theta1) * np.cos(theta2) + \
            L3 * np.sin(theta1) * np.cos(theta2 + theta3)
        z = L2 * np.sin(theta2) + L3 * np.sin(theta2 + theta3)
        
        return np.array([x, y, z])
    
    def inverse_kinematics_leg_numerical(self, leg_name: str, 
                                          target_pos_body: np.ndarray,
                                          model: Optional['mujoco.MjModel'] = None,
                                          data: Optional['mujoco.MjData'] = None,
                                          leg_joint_ids: Optional[list] = None) -> np.ndarray:
        """
        基于MuJoCo Jacobian的数值逆运动学（确保与物理模型一致）
        
        Args:
            leg_name: 腿名称
            target_pos_body: 目标位置（躯干坐标系）[3]
            model: MuJoCo模型
            data: MuJoCo数据
            leg_joint_ids: 该腿的关节ID列表 [3]
            
        Returns:
            np.ndarray: 关节角度 [3]
        """
        import mujoco
        
        if model is None or data is None:
            # 回退到解析解
            target_pos_leg = self.body_to_leg_frame(leg_name, target_pos_body)
            return self.inverse_kinematics_leg(leg_name, target_pos_leg)
        
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg_name.lower()}_tip")
        if body_id < 0:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg{['RF','RM','RR','LF','LM','LR'].index(leg_name)+1}_tip")
        
        if body_id < 0 or leg_joint_ids is None:
            target_pos_leg = self.body_to_leg_frame(leg_name, target_pos_body)
            return self.inverse_kinematics_leg(leg_name, target_pos_leg)
        
        # 转换目标位置到世界坐标系
        torso_pos = data.qpos[:3].copy()
        torso_quat = data.qpos[3:7].copy()
        R = np.zeros((9,))
        mujoco.mju_quat2Mat(R, torso_quat)
        R = R.reshape(3, 3)
        target_world = torso_pos + R @ target_pos_body
        
        # 获取当前关节角度
        current_angles = np.array([data.qpos[model.jnt_qposadr[jid]] for jid in leg_joint_ids])
        
        # 牛顿迭代
        for iteration in range(20):
            # 获取当前足端位置
            current_pos = data.xpos[body_id].copy()
            
            # 计算误差
            error = target_world - current_pos
            if np.linalg.norm(error) < 1e-5:
                break
            
            # 计算Jacobian
            jac_pos = np.zeros((3, model.nv))
            jac_rot = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jac_pos, jac_rot, body_id)
            
            # 提取相关列
            jac = np.zeros((3, 3))
            for j, jid in enumerate(leg_joint_ids):
                jac[:, j] = jac_pos[:, model.jnt_dofadr[jid]]
            
            # 阻尼最小二乘
            lambda_damp = 0.01
            jac_t = jac.T
            delta = jac_t @ np.linalg.inv(jac @ jac_t + lambda_damp * np.eye(3)) @ error
            
            # 更新关节角度
            current_angles += delta
            current_angles = np.clip(current_angles, -np.pi, np.pi)
            
            # 更新MuJoCo状态
            for j, jid in enumerate(leg_joint_ids):
                data.qpos[model.jnt_qposadr[jid]] = current_angles[j]
            mujoco.mj_forward(model, data)
        
        return current_angles
    
    def inverse_kinematics_leg(self, leg_name: str,
                                target_pos: np.ndarray,
                                current_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        单腿逆运动学（增强数值稳定性版本）
        
        修复问题:
        1. 对 arccos 参数进行 clip，防止超出 [-1, 1] 产生 NaN
        2. 对 d 添加下限保护，确保 d >= |L2 - L3| + epsilon
        3. 处理 x=y=0 的奇异情况
        
        Args:
            leg_name: 腿名称
            target_pos: 目标位置（腿坐标系）[3]
            current_angles: 当前关节角度（用于选择解）[3]
            
        Returns:
            np.ndarray: 关节角度 [3]
        """
        x, y, z = target_pos
        L1 = self.config.coxa_length
        L2 = self.config.femur_length
        L3 = self.config.tibia_length
        
        # === FIX 1: 处理 x=y=0 的奇异情况 ===
        # 当 x 和 y 都接近 0 时，arctan2 不稳定，coxa 角度不确定
        # 添加微小偏移避免奇异
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            x = 1e-6
        
        # 计算coxa角度（水平旋转）
        theta1 = np.arctan2(y, x)
        
        # 转换到 femur-tibia 平面
        r = np.sqrt(x**2 + y**2) - L1
        d = np.sqrt(r**2 + z**2)
        
        # === FIX 2: 对 d 添加双向软限制 ===
        # d 必须在 [|L2 - L3| + eps, L2 + L3 - eps] 范围内
        # 否则 arccos 的参数会超出 [-1, 1]，产生 NaN
        d_min = abs(L2 - L3) + 0.001  # 0.021 m for this robot
        d_max = L2 + L3 - 0.001       # 0.184 m for this robot
        d = np.clip(d, d_min, d_max)
        
        # === FIX 3: 对两个 arccos 参数都进行 clip ===
        # 原始代码只对 cos_theta3 进行了 clip，但 beta 的 arccos 参数也需要 clip
        
        # 使用余弦定理计算 tibia 角度 (theta3)
        cos_theta3 = (d**2 - L2**2 - L3**2) / (2 * L2 * L3)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3 = -np.arccos(cos_theta3)
        
        # 计算 femur 角度 (theta2)
        alpha = np.arctan2(z, r)
        
        # beta 的 arccos 参数也必须 clip！这是产生 NaN 的关键原因
        beta_arg = (L2**2 + d**2 - L3**2) / (2 * L2 * d)
        beta_arg = np.clip(beta_arg, -1.0, 1.0)
        beta = np.arccos(beta_arg)
        theta2 = alpha + beta
        
        # 应用关节方向符号（逆运动学结果需要乘以符号）
        signs = self.config.joint_signs[leg_name]
        joint_angles = np.array([theta1, theta2, theta3]) * signs
        
        return joint_angles
    
    def check_reachability(self, leg_name: str, target_pos: np.ndarray) -> Tuple[bool, float]:
        """
        检查目标位置是否在腿的工作空间内
        
        Args:
            leg_name: 腿名称
            target_pos: 目标位置（腿坐标系）[3]
            
        Returns:
            Tuple[bool, float]: (是否可达, 到可达边界的距离)
        """
        x, y, z = target_pos
        L1 = self.config.coxa_length
        L2 = self.config.femur_length
        L3 = self.config.tibia_length
        
        r = np.sqrt(x**2 + y**2) - L1
        d = np.sqrt(r**2 + z**2)
        
        d_min = abs(L2 - L3)
        d_max = L2 + L3
        
        reachable = d_min <= d <= d_max
        margin = min(d - d_min, d_max - d)
        
        return reachable, margin
    
    def body_to_leg_frame(self, leg_name: str, 
                          body_point: np.ndarray) -> np.ndarray:
        """
        将躯干坐标系中的点转换到腿坐标系
        
        Args:
            leg_name: 腿名称
            body_point: 躯干坐标系中的点 [3]
            
        Returns:
            np.ndarray: 腿坐标系中的点 [3]
        """
        mount_pos = self.config.mount_positions[leg_name]
        R = self.leg_transforms[leg_name]
        
        # 平移并旋转
        leg_point = R.T @ (body_point - mount_pos)
        return leg_point
    
    def leg_to_body_frame(self, leg_name: str,
                          leg_point: np.ndarray) -> np.ndarray:
        """
        将腿坐标系中的点转换到躯干坐标系
        
        Args:
            leg_name: 腿名称
            leg_point: 腿坐标系中的点 [3]
            
        Returns:
            np.ndarray: 躯干坐标系中的点 [3]
        """
        mount_pos = self.config.mount_positions[leg_name]
        R = self.leg_transforms[leg_name]
        
        body_point = R @ leg_point + mount_pos
        return body_point
    
    def compute_foot_positions_body_frame(self,
                                          joint_angles: np.ndarray) -> np.ndarray:
        """
        计算所有足底在躯干坐标系中的位置
        
        Args:
            joint_angles: 所有关节角度 [18]
            
        Returns:
            np.ndarray: 足底位置 [6, 3]
        """
        foot_positions = np.zeros((6, 3))
        leg_names = ['RF', 'RM', 'RR', 'LF', 'LM', 'LR']
        
        for i, leg_name in enumerate(leg_names):
            leg_joints = joint_angles[i*3:(i+1)*3]
            foot_leg_frame = self.forward_kinematics_leg(leg_name, leg_joints)
            foot_positions[i] = self.leg_to_body_frame(leg_name, foot_leg_frame)
        
        return foot_positions
    
    def compute_jacobians(self, leg_name: str,
                          joint_angles: np.ndarray) -> np.ndarray:
        """
        计算单腿雅可比矩阵
        
        Args:
            leg_name: 腿名称
            joint_angles: 关节角度 [3]
            
        Returns:
            np.ndarray: 雅可比矩阵 [3, 3]
        """
        theta1, theta2, theta3 = joint_angles
        L1 = self.config.coxa_length
        L2 = self.config.femur_length
        L3 = self.config.tibia_length
        
        s1, c1 = np.sin(theta1), np.cos(theta1)
        s2, c2 = np.sin(theta2), np.cos(theta2)
        s23, c23 = np.sin(theta2 + theta3), np.cos(theta2 + theta3)
        
        # 雅可比矩阵
        J = np.array([
            [-L1*s1 - L2*s1*c2 - L3*s1*c23, -L2*c1*s2 - L3*c1*s23, -L3*c1*s23],
            [L1*c1 + L2*c1*c2 + L3*c1*c23, -L2*s1*s2 - L3*s1*s23, -L3*s1*s23],
            [0, -L2*c2 - L3*c23, -L3*c23]
        ])
        
        return J
    
    def compute_support_polygon(self, foot_positions: np.ndarray,
                                 contact_states: np.ndarray) -> np.ndarray:
        """
        计算支撑多边形
        
        Args:
            foot_positions: 足底位置 [6, 3]
            contact_states: 接触状态 [6]
            
        Returns:
            np.ndarray: 支撑多边形顶点 [N, 2]
        """
        support_feet = foot_positions[contact_states > 0]
        if len(support_feet) < 3:
            return np.array([])
        
        # 计算凸包
        from scipy.spatial import ConvexHull
        points_2d = support_feet[:, :2]
        
        try:
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        except:
            return points_2d
    
    def check_static_stability(self, foot_positions: np.ndarray,
                                contact_states: np.ndarray,
                                com_position: np.ndarray) -> bool:
        """
        检查静态稳定性
        
        Args:
            foot_positions: 足底位置 [6, 3]
            contact_states: 接触状态 [6]
            com_position: 质心位置 [3]
            
        Returns:
            bool: 是否稳定
        """
        support_polygon = self.compute_support_polygon(foot_positions, contact_states)
        
        if len(support_polygon) < 3:
            return False
        
        # 检查COM投影是否在支撑多边形内
        com_2d = com_position[:2]
        
        # 使用射线法判断点是否在多边形内
        n = len(support_polygon)
        inside = False
        j = n - 1
        
        for i in range(n):
            xi, yi = support_polygon[i]
            xj, yj = support_polygon[j]
            
            if ((yi > com_2d[1]) != (yj > com_2d[1])) and \
               (com_2d[0] < (xj - xi) * (com_2d[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside

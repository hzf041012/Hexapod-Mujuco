"""
Balance Controller Module
Implements posture stabilization, height control, body tilt compensation, etc.
Enhanced for stair climbing with StairCompensator

修复内容:
1. height_pid 增益从过大值 (kp=10, kd=1) 调整到合理范围
2. 添加 adjustments 输出范围限制
3. 添加 NaN/Inf 安全检查
4. 积分项添加更严格的限制，防止积分饱和
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BalanceParameters:
    """Balance control parameters"""
    # PID parameters - 修复: 降低 height_pid 增益，防止过大输出
    kp_roll: float = 5.0
    ki_roll: float = 0.1
    kd_roll: float = 0.5
    
    kp_pitch: float = 5.0
    ki_pitch: float = 0.1
    kd_pitch: float = 0.5
    
    # FIX: 降低 height_pid 增益。原值 (kp=10, kd=1) 在 dt=0.01s 时微分项过大
    # 新值: kp=2.0, kd=0.1, 更温和的高度调整
    kp_height: float = 2.0
    ki_height: float = 0.2
    kd_height: float = 0.1
    
    # Target values
    target_roll: float = 0.0
    target_pitch: float = 0.0
    target_height: float = 0.12
    
    # Limits
    max_roll_correction: float = 0.15  # Max roll correction (rad)
    max_pitch_correction: float = 0.15  # Max pitch correction (rad)
    max_height_change: float = 0.03   # Max height change (m)
    
    # Stair compensation parameters
    stair_slope_threshold: float = 0.02  # Slope threshold for stair detection (rad)
    front_rear_height_diff_threshold: float = 0.02  # Front-rear height diff threshold (m)


class PIDController:
    """PID Controller"""
    
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: Optional[Tuple[float, float]] = None):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limit: Output limit (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = 1.0
        
    def reset(self):
        """Reset controller"""
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, error: float, dt: float) -> float:
        """
        Update controller
        
        Args:
            error: Error
            dt: Time step
            
        Returns:
            float: Control output
        """
        # FIX: 添加 dt 安全检查
        if dt <= 0 or not np.isfinite(dt):
            dt = 0.001
        
        # FIX: 添加 error 安全检查
        if not np.isfinite(error):
            error = 0.0
        
        # Integral
        self.integral += error * dt
        # FIX: 更严格的积分限制，防止积分饱和
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Derivative
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Output limit
        if self.output_limit is not None:
            output = np.clip(output, self.output_limit[0], self.output_limit[1])
        
        # FIX: 确保输出是有限值
        if not np.isfinite(output):
            output = 0.0
        
        return output


class BalanceController:
    """
    Balance Controller
    Maintains body balance by adjusting foot end positions
    Enhanced with stair slope compensation
    
    修复: 降低 PID 增益，添加输出限制和 NaN 检查
    """
    
    def __init__(self, params: Optional[BalanceParameters] = None):
        """
        Initialize balance controller
        
        Args:
            params: Balance control parameters
        """
        self.params = params or BalanceParameters()
        
        # Create PID controllers
        self.roll_pid = PIDController(
            self.params.kp_roll,
            self.params.ki_roll,
            self.params.kd_roll,
            (-self.params.max_roll_correction, self.params.max_roll_correction)
        )
        
        self.pitch_pid = PIDController(
            self.params.kp_pitch,
            self.params.ki_pitch,
            self.params.kd_pitch,
            (-self.params.max_pitch_correction, self.params.max_pitch_correction)
        )
        
        self.height_pid = PIDController(
            self.params.kp_height,
            self.params.ki_height,
            self.params.kd_height,
            (-self.params.max_height_change, self.params.max_height_change)
        )
        
    def reset(self):
        """Reset controller"""
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.height_pid.reset()
    
    def compute_body_adjustment(self, roll: float, pitch: float, height: float,
                                 dt: float) -> Tuple[float, float, float]:
        """
        Compute body posture adjustment
        
        Args:
            roll: Current roll angle (rad)
            pitch: Current pitch angle (rad)
            height: Current height (m)
            dt: Time step
            
        Returns:
            Tuple[float, float, float]: (roll_correction, pitch_correction, height_correction)
        """
        # FIX: 安全检查输入值
        if not np.isfinite(roll):
            roll = 0.0
        if not np.isfinite(pitch):
            pitch = 0.0
        if not np.isfinite(height):
            height = self.params.target_height
        
        # Compute errors
        roll_error = self.params.target_roll - roll
        pitch_error = self.params.target_pitch - pitch
        height_error = self.params.target_height - height
        
        # PID control
        roll_correction = self.roll_pid.update(roll_error, dt)
        pitch_correction = self.pitch_pid.update(pitch_error, dt)
        height_correction = self.height_pid.update(height_error, dt)
        
        return roll_correction, pitch_correction, height_correction
    
    def compute_foot_adjustments(self, roll: float, pitch: float, height: float,
                                  foot_positions: np.ndarray,
                                  contact_states: np.ndarray,
                                  dt: float,
                                  stair_slope: float = 0.0) -> np.ndarray:
        """
        Compute foot position adjustments (Enhanced with stair slope compensation)
        
        修复:
        - 添加输出范围限制
        - 添加 NaN/Inf 安全检查
        - 当没有足端接触时返回零调整
        
        Args:
            roll: Current roll angle
            pitch: Current pitch angle
            height: Current height
            foot_positions: Current foot end positions [6, 3]
            contact_states: Contact states [6]
            dt: Time step
            stair_slope: Detected stair slope (rad), 0 if no stair
            
        Returns:
            np.ndarray: Foot position adjustments [6, 3]
        """
        # FIX: 安全检查输入
        if foot_positions is None or not np.all(np.isfinite(foot_positions)):
            return np.zeros((6, 3))
        if contact_states is None:
            return np.zeros((6, 3))
        
        roll_corr, pitch_corr, height_corr = self.compute_body_adjustment(
            roll, pitch, height, dt)
        
        adjustments = np.zeros((6, 3))
        
        # FIX: 如果没有足端接触地面，不做调整（防止空中调整导致不稳定）
        if np.sum(contact_states) == 0:
            return adjustments
        
        # Calculate front-rear height difference for stair detection
        front_contacts = [i for i in [0, 3] if contact_states[i] > 0]
        rear_contacts = [i for i in [2, 5] if contact_states[i] > 0]
        
        front_z = np.mean([foot_positions[i][2] for i in front_contacts]) if front_contacts else 0.0
        rear_z = np.mean([foot_positions[i][2] for i in rear_contacts]) if rear_contacts else 0.0
        z_diff = front_z - rear_z
        
        # Check if we are on stairs (significant front-rear height difference)
        is_on_stair = abs(z_diff) > self.params.front_rear_height_diff_threshold
        
        for i in range(6):
            if contact_states[i] > 0:
                # Stance legs: adjust position based on body tilt
                # Roll adjustment: left/right legs move in opposite directions
                if i < 3:  # Right legs
                    adjustments[i, 2] += roll_corr * 0.5
                else:  # Left legs
                    adjustments[i, 2] -= roll_corr * 0.5
                
                # Pitch adjustment: front/rear legs move in opposite directions
                if i in [0, 3]:  # Front legs
                    adjustments[i, 2] -= pitch_corr * 0.5
                elif i in [2, 5]:  # Rear legs
                    adjustments[i, 2] += pitch_corr * 0.5
                
                # Height adjustment
                adjustments[i, 2] += height_corr
                
                # === Stair slope compensation ===
                if is_on_stair:
                    # When front legs are on higher step, reduce front foot height
                    # and increase rear foot height for better stability
                    if i in [0, 3]:  # Front legs
                        # Front legs: negative compensation (lower foot position)
                        adjustments[i, 2] -= z_diff * 0.3
                    elif i in [2, 5]:  # Rear legs
                        # Rear legs: positive compensation (raise foot position slightly)
                        adjustments[i, 2] += z_diff * 0.2
                    
                    # Additional slope compensation based on detected stair slope
                    if abs(stair_slope) > self.params.stair_slope_threshold:
                        slope_compensation = stair_slope * 0.1
                        adjustments[i, 2] += slope_compensation
        
        # FIX: 限制每个调整量的大小，防止过大调整
        max_adjustment_z = 0.05  # 最大 Z 方向调整 5cm
        for i in range(6):
            adjustments[i, 2] = np.clip(adjustments[i, 2], -max_adjustment_z, max_adjustment_z)
        
        # FIX: 最终安全检查 - 确保没有 NaN/Inf
        if not np.all(np.isfinite(adjustments)):
            adjustments = np.zeros((6, 3))
        
        return adjustments
    
    def compute_support_polygon_center(self, foot_positions: np.ndarray,
                                        contact_states: np.ndarray) -> np.ndarray:
        """
        Compute support polygon center
        
        Args:
            foot_positions: Foot end positions [6, 3]
            contact_states: Contact states [6]
            
        Returns:
            np.ndarray: Support center [3]
        """
        support_feet = foot_positions[contact_states > 0]
        if len(support_feet) == 0:
            return np.zeros(3)
        return np.mean(support_feet, axis=0)
    
    def compute_stability_margin(self, foot_positions: np.ndarray,
                                  contact_states: np.ndarray,
                                  com_position: np.ndarray) -> float:
        """
        Compute static stability margin
        
        Args:
            foot_positions: Foot end positions [6, 3]
            contact_states: Contact states [6]
            com_position: Center of mass position [3]
            
        Returns:
            float: Stability margin (minimum distance to support boundary)
        """
        support_feet = foot_positions[contact_states > 0]
        if len(support_feet) < 3:
            return 0.0
        
        com_2d = com_position[:2]
        min_distance = float('inf')
        
        # Compute distance to each edge
        n = len(support_feet)
        for i in range(n):
            p1 = support_feet[i, :2]
            p2 = support_feet[(i+1) % n, :2]
            
            # Point to line segment distance
            line_vec = p2 - p1
            point_vec = com_2d - p1
            
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                distance = np.linalg.norm(point_vec)
            else:
                t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
                projection = p1 + t * line_vec
                distance = np.linalg.norm(com_2d - projection)
            
            min_distance = min(min_distance, distance)
        
        return min_distance


class ComplianceController:
    """
    Compliance Controller
    Implements foot end compliance to adapt to uneven terrain
    """
    
    def __init__(self, stiffness: float = 100.0, damping: float = 10.0):
        """
        Initialize compliance controller
        
        Args:
            stiffness: Stiffness coefficient (N/m)
            damping: Damping coefficient (N.s/m)
        """
        self.stiffness = stiffness
        self.damping = damping
        
    def compute_compliant_force(self, position_error: np.ndarray,
                                 velocity: np.ndarray) -> np.ndarray:
        """
        Compute compliant force
        
        Args:
            position_error: Position error [3]
            velocity: Velocity [3]
            
        Returns:
            np.ndarray: Compliant force [3]
        """
        force = -self.stiffness * position_error - self.damping * velocity
        return force
    
    def adapt_stiffness(self, terrain_stiffness: float):
        """
        Adapt stiffness based on terrain stiffness
        
        Args:
            terrain_stiffness: Terrain stiffness estimate
        """
        # Increase stiffness on soft terrain, decrease on hard terrain
        self.stiffness = 100.0 * (1.0 / (1.0 + terrain_stiffness))


class StairCompensator:
    """
    Stair Compensator
    Detects stair slope through front/rear leg height differences,
    automatically adjusts body pitch angle to match stair slope,
    provides forward lean compensation during climbing.
    """
    
    def __init__(self, max_pitch_adjustment: float = 0.2,
                 body_length: float = 0.2):
        """
        Initialize stair compensator
        
        Args:
            max_pitch_adjustment: Maximum pitch angle adjustment (rad)
            body_length: Distance between front and rear legs (m)
        """
        self.max_pitch_adjustment = max_pitch_adjustment
        self.body_length = body_length
        
        # State variables
        self.detected_slope = 0.0
        self.target_pitch = 0.0
        self.pitch_error_integral = 0.0
        self.pitch_error_prev = 0.0
        
        # PID parameters for pitch compensation
        self.kp = 2.0
        self.ki = 0.1
        self.kd = 0.3
        
        # Forward lean compensation
        self.forward_lean_angle = 0.05  # Default forward lean (rad) ~ 2.87 degrees
        
        # Smoothing
        self.smoothed_slope = 0.0
        self.slope_alpha = 0.1  # Exponential smoothing factor
    
    def detect_slope(self, front_leg_heights: np.ndarray,
                     rear_leg_heights: np.ndarray) -> float:
        """
        Detect stair slope through front/rear leg height differences
        
        Args:
            front_leg_heights: Front leg foot end heights [2] (right, left)
            rear_leg_heights: Rear leg foot end heights [2]
            
        Returns:
            float: Detected slope (rad)
        """
        front_avg = np.mean(front_leg_heights)
        rear_avg = np.mean(rear_leg_heights)
        
        height_diff = front_avg - rear_avg
        # Clamp height diff to avoid extreme slope estimates
        height_diff = np.clip(height_diff, -0.3, 0.3)
        
        self.detected_slope = np.arctan2(height_diff, self.body_length)
        
        # Exponential smoothing for stable slope estimate
        self.smoothed_slope = self.slope_alpha * self.detected_slope + \
                              (1 - self.slope_alpha) * self.smoothed_slope
        
        return self.smoothed_slope
    
    def compute_pitch_compensation(self, current_pitch: float,
                                    front_leg_heights: np.ndarray,
                                    rear_leg_heights: np.ndarray,
                                    dt: float) -> float:
        """
        Compute pitch angle compensation
        
        When stairs are detected:
        - Body leans slightly forward to match stair slope
        - Forward lean helps front legs climb steps more easily
        
        Args:
            current_pitch: Current pitch angle (rad)
            front_leg_heights: Front leg foot end heights
            rear_leg_heights: Rear leg foot end heights
            dt: Time step
            
        Returns:
            float: Pitch angle compensation amount
        """
        # Detect slope
        slope = self.detect_slope(front_leg_heights, rear_leg_heights)
        
        # Target pitch: match stair slope + extra forward lean compensation
        # Forward lean helps with climbing
        if slope > 0.05:  # Climbing upward
            forward_lean = self.forward_lean_angle
        elif slope < -0.05:  # Going downward
            forward_lean = -self.forward_lean_angle * 0.5
        else:
            forward_lean = 0.0
        
        self.target_pitch = slope + forward_lean
        
        # PID control
        error = self.target_pitch - current_pitch
        self.pitch_error_integral += error * dt
        self.pitch_error_integral = np.clip(self.pitch_error_integral, -0.5, 0.5)
        
        derivative = (error - self.pitch_error_prev) / dt if dt > 0 else 0
        self.pitch_error_prev = error
        
        compensation = (self.kp * error +
                       self.ki * self.pitch_error_integral +
                       self.kd * derivative)
        
        return float(np.clip(compensation, -self.max_pitch_adjustment, self.max_pitch_adjustment))
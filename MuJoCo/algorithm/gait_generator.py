"""
Gait Generator Module
Implements multiple hexapod gaits: Tripod, Wave, Ripple Gait
Enhanced for stair climbing with StairAdaptiveGait

修复内容:
1. 所有步态的 get_foot_trajectory 添加足端位置数值保护
2. 限制 target_vel 的大小，防止过大速度指令
3. 确保 swing_progress / stance_progress 的计算安全
4. 统一 Z 方向处理：支撑相保持 home 高度，摆动相正确抬腿
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GaitPhase(Enum):
    """Gait phase"""
    STANCE = 0    # Stance phase
    SWING = 1     # Swing phase


@dataclass
class GaitParameters:
    """Gait parameters"""
    step_height: float = 0.03      # Step height (m)
    step_length: float = 0.08      # Step length (m)
    cycle_time: float = 1.0        # Cycle time (s)
    duty_factor: float = 0.5       # Duty factor
    leg_offset: float = 0.0        # Leg phase offset


class GaitGenerator:
    """
    Base Gait Generator
    """
    
    def __init__(self, params: Optional[GaitParameters] = None):
        """
        Initialize gait generator
        
        Args:
            params: Gait parameters
        """
        self.params = params or GaitParameters()
        self.time = 0.0
        
    def update(self, dt: float, velocity_command: np.ndarray):
        """
        Update gait state
        
        Args:
            dt: Time step
            velocity_command: Velocity command [vx, vy, omega]
        """
        self.time += dt
        
    def get_foot_trajectory(self, leg_idx: int, 
                            home_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trajectory for specified foot
        
        Args:
            leg_idx: Leg index (0-5)
            home_position: Foot home position [3]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (target position, target velocity)
        """
        raise NotImplementedError
    
    def get_leg_phase(self, leg_idx: int) -> float:
        """
        Get leg phase (0-1)
        
        Args:
            leg_idx: Leg index
            
        Returns:
            float: Phase value
        """
        raise NotImplementedError
    
    def is_stance_phase(self, leg_idx: int) -> bool:
        """
        Check if leg is in stance phase
        
        Args:
            leg_idx: Leg index
            
        Returns:
            bool: Whether in stance phase
        """
        raise NotImplementedError


def _compute_stance_trajectory(home_position, stance_progress, stride, vy, cycle_time, duty_factor):
    """
    统一支撑相轨迹计算（防止代码重复和一致性问题）
    
    Args:
        home_position:  home 位置 [3]
        stance_progress: 支撑相进度 [0, 1]
        stride: 步长
        vy: Y方向速度
        cycle_time: 周期时间
        duty_factor: 占空比
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (target_pos, target_vel)
    """
    x_offset = stride * (0.5 - stance_progress)
    y_offset = vy * cycle_time * duty_factor * (0.5 - stance_progress)
    
    target_pos = home_position.copy()
    target_pos[0] += x_offset
    target_pos[1] += y_offset
    # 支撑相：保持 home 高度（足端在地面上）
    target_pos[2] = home_position[2]
    
    vx_speed = -stride / (cycle_time * duty_factor) if cycle_time * duty_factor > 1e-6 else 0.0
    vy_speed = -vy if abs(vy) > 1e-6 else 0.0
    target_vel = np.array([vx_speed, vy_speed, 0.0])
    
    return target_pos, target_vel


def _compute_swing_trajectory(home_position, swing_progress, stride, vy, step_height, cycle_time, duty_factor):
    """
    统一摆动相轨迹计算（防止代码重复和一致性问题）
    
    修复: 
    - 确保 vz 计算不会除零
    - 限制最大速度
    
    Args:
        home_position: home 位置 [3]
        swing_progress: 摆动相进度 [0, 1]
        stride: 步长
        vy: Y方向速度
        step_height: 步高
        cycle_time: 周期时间
        duty_factor: 占空比
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (target_pos, target_vel)
    """
    x_offset = stride * (swing_progress - 0.5)
    y_offset = vy * cycle_time * (1 - duty_factor) * (swing_progress - 0.5)
    
    # Z方向：正弦抬腿
    z_height = step_height * np.sin(swing_progress * np.pi)
    
    target_pos = home_position.copy()
    target_pos[0] += x_offset
    target_pos[1] += y_offset
    # 摆动相：向上抬腿（home_position[2] 为负值，加上正值使足端更高/更接近0）
    target_pos[2] = home_position[2] + z_height
    
    # 速度计算（防止除零）
    swing_duration = cycle_time * (1 - duty_factor)
    if swing_duration > 1e-6:
        vx_speed = stride / swing_duration
        vz = step_height * np.pi * np.cos(swing_progress * np.pi) / swing_duration
    else:
        vx_speed = 0.0
        vz = 0.0
    
    vy_speed = vy if abs(vy) > 1e-6 else 0.0
    
    target_vel = np.array([vx_speed, vy_speed, -vz])
    
    # 限制最大速度（防止过大速度值）
    max_vel = 5.0  # m/s
    vel_norm = np.linalg.norm(target_vel)
    if vel_norm > max_vel:
        target_vel = target_vel * (max_vel / vel_norm)
    
    return target_pos, target_vel


class TripodGait(GaitGenerator):
    """
    Tripod Gait
    Legs are divided into two groups that alternate: {RF, LR, LM} and {LF, RR, RM}
    Fastest gait but less stable
    """
    
    # Tripod grouping
    GROUP_A = [0, 3, 4]  # RF, LR, LM
    GROUP_B = [2, 5, 1]  # RR, LF, RM
    
    def __init__(self, params: Optional[GaitParameters] = None):
        super().__init__(params)
        # Tripod gait duty factor is typically 0.5
        self.params.duty_factor = 0.5
        
    def get_leg_phase(self, leg_idx: int) -> float:
        """Get leg phase"""
        # Legs in the same group share the same phase
        if leg_idx in self.GROUP_A:
            phase = (self.time / self.params.cycle_time) % 1.0
        else:
            phase = ((self.time / self.params.cycle_time) + 0.5) % 1.0
        return phase
    
    def is_stance_phase(self, leg_idx: int) -> bool:
        """Check if in stance phase"""
        phase = self.get_leg_phase(leg_idx)
        return phase < self.params.duty_factor
    
    def get_foot_trajectory(self, leg_idx: int,
                            home_position: np.ndarray,
                            velocity_command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute foot trajectory（修复版本）
        
        修复:
        - 使用统一的 _compute_stance_trajectory 和 _compute_swing_trajectory
        - 根据 vx 计算实际步长（velocity=0 时 stride=0）
        - 确保进度计算安全（防止除零）
        """
        phase = self.get_leg_phase(leg_idx)
        vx = float(velocity_command[0])
        vy = float(velocity_command[1]) if len(velocity_command) > 1 else 0.0
        
        # FIX: 根据速度计算实际步长，而不是使用固定的 step_length
        stride = min(self.params.step_length, abs(vx) * self.params.cycle_time)
        
        if self.is_stance_phase(leg_idx):
            # Stance phase
            # FIX: 防止 duty_factor 接近 0 导致除零
            duty = max(self.params.duty_factor, 0.1)
            stance_progress = phase / duty
            stance_progress = np.clip(stance_progress, 0.0, 1.0)
            
            return _compute_stance_trajectory(
                home_position, stance_progress,
                stride, vy,
                self.params.cycle_time, duty
            )
        else:
            # Swing phase
            swing_denom = 1.0 - self.params.duty_factor
            swing_denom = max(swing_denom, 0.1)  # FIX: 防止除零
            swing_progress = (phase - self.params.duty_factor) / swing_denom
            swing_progress = np.clip(swing_progress, 0.0, 1.0)
            
            return _compute_swing_trajectory(
                home_position, swing_progress,
                stride, vy,
                self.params.step_height,
                self.params.cycle_time, self.params.duty_factor
            )


class WaveGait(GaitGenerator):
    """
    Wave Gait
    Six legs move sequentially, forming a wave
    Most stable gait but slowest
    """
    
    # Leg phase offsets (evenly spaced)
    LEG_PHASE_OFFSETS = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6]
    
    def __init__(self, params: Optional[GaitParameters] = None):
        super().__init__(params)
        # Wave gait duty factor is typically 5/6
        self.params.duty_factor = 5/6
        
    def get_leg_phase(self, leg_idx: int) -> float:
        """Get leg phase"""
        offset = self.LEG_PHASE_OFFSETS[leg_idx]
        phase = ((self.time / self.params.cycle_time) + offset) % 1.0
        return phase
    
    def is_stance_phase(self, leg_idx: int) -> bool:
        """Check if in stance phase"""
        phase = self.get_leg_phase(leg_idx)
        return phase < self.params.duty_factor
    
    def get_foot_trajectory(self, leg_idx: int,
                            home_position: np.ndarray,
                            velocity_command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute foot trajectory（修复版本）"""
        phase = self.get_leg_phase(leg_idx)
        vx = float(velocity_command[0])
        vy = float(velocity_command[1]) if len(velocity_command) > 1 else 0.0
        
        # FIX: 根据速度计算实际步长
        stride = min(self.params.step_length, abs(vx) * self.params.cycle_time)
        
        if self.is_stance_phase(leg_idx):
            duty = max(self.params.duty_factor, 0.1)
            stance_progress = phase / duty
            stance_progress = np.clip(stance_progress, 0.0, 1.0)
            
            return _compute_stance_trajectory(
                home_position, stance_progress,
                stride, vy,
                self.params.cycle_time, duty
            )
        else:
            swing_denom = 1.0 - self.params.duty_factor
            swing_denom = max(swing_denom, 0.1)
            swing_progress = (phase - self.params.duty_factor) / swing_denom
            swing_progress = np.clip(swing_progress, 0.0, 1.0)
            
            return _compute_swing_trajectory(
                home_position, swing_progress,
                stride, vy,
                self.params.step_height,
                self.params.cycle_time, self.params.duty_factor
            )


class RippleGait(GaitGenerator):
    """
    Ripple Gait
    Between Tripod and Wave gait
    Two legs swing at a time, four legs support
    Balances speed and stability
    """
    
    # Leg phase offsets
    LEG_PHASE_OFFSETS = [0.0, 0.5, 1/3, 5/6, 1/6, 2/3]
    
    def __init__(self, params: Optional[GaitParameters] = None):
        super().__init__(params)
        # Ripple gait duty factor is typically 2/3
        self.params.duty_factor = 2/3
        
    def get_leg_phase(self, leg_idx: int) -> float:
        """Get leg phase"""
        offset = self.LEG_PHASE_OFFSETS[leg_idx]
        phase = ((self.time / self.params.cycle_time) + offset) % 1.0
        return phase
    
    def is_stance_phase(self, leg_idx: int) -> bool:
        """Check if in stance phase"""
        phase = self.get_leg_phase(leg_idx)
        return phase < self.params.duty_factor
    
    def get_foot_trajectory(self, leg_idx: int,
                            home_position: np.ndarray,
                            velocity_command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute foot trajectory（修复版本）"""
        phase = self.get_leg_phase(leg_idx)
        vx = float(velocity_command[0])
        vy = float(velocity_command[1]) if len(velocity_command) > 1 else 0.0
        
        # FIX: 根据速度计算实际步长
        stride = min(self.params.step_length, abs(vx) * self.params.cycle_time)
        
        if self.is_stance_phase(leg_idx):
            duty = max(self.params.duty_factor, 0.1)
            stance_progress = phase / duty
            stance_progress = np.clip(stance_progress, 0.0, 1.0)
            
            return _compute_stance_trajectory(
                home_position, stance_progress,
                stride, vy,
                self.params.cycle_time, duty
            )
        else:
            swing_denom = 1.0 - self.params.duty_factor
            swing_denom = max(swing_denom, 0.1)
            swing_progress = (phase - self.params.duty_factor) / swing_denom
            swing_progress = np.clip(swing_progress, 0.0, 1.0)
            
            return _compute_swing_trajectory(
                home_position, swing_progress,
                stride, vy,
                self.params.step_height,
                self.params.cycle_time, self.params.duty_factor
            )


class AdaptiveGait(GaitGenerator):
    """
    Adaptive Gait
    Dynamically adjusts gait parameters based on terrain and stability requirements
    """
    
    def __init__(self, params: Optional[GaitParameters] = None):
        super().__init__(params)
        self.base_cycle_time = self.params.cycle_time
        self.stability_margin = 0.0
        
    def adapt_to_terrain(self, terrain_slope: float, terrain_roughness: float):
        """
        Adapt to terrain
        
        Args:
            terrain_slope: Terrain slope
            terrain_roughness: Terrain roughness
        """
        # Larger slope means slower and more stable gait
        slope_factor = 1.0 + abs(terrain_slope)
        roughness_factor = 1.0 + terrain_roughness * 2
        
        self.params.cycle_time = self.base_cycle_time * slope_factor * roughness_factor
        self.params.step_height = 0.03 + terrain_roughness * 0.02
        
    def adapt_to_stability(self, stability_margin: float):
        """
        Adjust based on stability margin
        
        Args:
            stability_margin: Stability margin
        """
        self.stability_margin = stability_margin
        
        # Reduce speed when stability margin is low
        if stability_margin < 0.02:
            self.params.cycle_time = self.base_cycle_time * 1.5
            self.params.step_length = 0.05
        else:
            self.params.cycle_time = self.base_cycle_time
            self.params.step_length = 0.08
    
    def get_leg_phase(self, leg_idx: int) -> float:
        """Use wave gait as base"""
        offset = WaveGait.LEG_PHASE_OFFSETS[leg_idx]
        phase = ((self.time / self.params.cycle_time) + offset) % 1.0
        return phase
    
    def is_stance_phase(self, leg_idx: int) -> bool:
        """Check if in stance phase"""
        phase = self.get_leg_phase(leg_idx)
        return phase < self.params.duty_factor
    
    def get_foot_trajectory(self, leg_idx: int,
                            home_position: np.ndarray,
                            velocity_command: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute foot trajectory"""
        # Reuse wave gait trajectory calculation
        wave_gait = WaveGait(self.params)
        wave_gait.time = self.time
        return wave_gait.get_foot_trajectory(leg_idx, home_position, velocity_command)


class StairAdaptiveGait(GaitGenerator):
    """
    Stair Adaptive Gait
    Inherits from GaitGenerator
    
    Automatically detects stairs through front/rear leg height differences and
    adjusts gait parameters:
    - Increases swing height for climbing
    - Shortens step length for stability
    - Slows cycle time for safety
    - Front legs lift higher to ascend steps
    - Rear legs provide stable support
    
    Uses Wave Gait as the base for maximum stability.
    """
    
    # Front legs: right front=0, left front=3
    # Rear legs: right rear=2, left rear=5
    FRONT_LEGS = [0, 3]
    REAR_LEGS = [2, 5]
    
    def __init__(self, params: Optional[GaitParameters] = None):
        super().__init__(params)
        
        # Save default flat terrain parameters
        self.default_params = GaitParameters(
            step_height=self.params.step_height,
            step_length=self.params.step_length,
            cycle_time=self.params.cycle_time,
            duty_factor=5/6  # Wave gait duty factor
        )
        
        # Climbing parameters
        self.climb_params = GaitParameters(
            step_height=0.06,    # Double the original step height
            step_length=0.05,    # Shorter steps for stability
            cycle_time=1.5,      # Slower for safety
            duty_factor=5/6      # Wave gait duty factor
        )
        
        # Use wave gait duty factor
        self.params.duty_factor = 5/6
        
        # Stair detection state
        self.is_stair_detected = False
        self.stair_height = 0.0
        self.front_leg_height = 0.0
        self.rear_leg_height = 0.0
        
        # Smooth transition parameters
        self.transition_factor = 0.0  # 0=flat, 1=climbing
        self.transition_rate = 0.5    # Transition rate per second
        
        # Stair detection threshold
        self.stair_detection_threshold = 0.03  # 3cm height difference
    
    def detect_stair(self, foot_positions: np.ndarray):
        """
        Detect stairs through front/rear leg height differences
        
        Args:
            foot_positions: Current foot end positions [6, 3]
        """
        # Calculate average height of front and rear legs
        front_z = np.mean([foot_positions[i][2] for i in self.FRONT_LEGS])
        rear_z = np.mean([foot_positions[i][2] for i in self.REAR_LEGS])
        
        self.front_leg_height = front_z
        self.rear_leg_height = rear_z
        
        # Detect height difference
        height_diff = abs(front_z - rear_z)
        self.stair_height = height_diff
        
        # If front-rear height difference exceeds threshold, consider stair detected
        self.is_stair_detected = height_diff > self.stair_detection_threshold
    
    def _lerp(self, a: float, b: float, t: float) -> float:
        """Linear interpolation"""
        return a + (b - a) * np.clip(t, 0.0, 1.0)
    
    def update(self, dt: float, velocity_command: np.ndarray,
               foot_positions: Optional[np.ndarray] = None):
        """
        Update gait state, including stair detection and parameter adaptation
        
        Args:
            dt: Time step
            velocity_command: Velocity command [vx, vy, omega]
            foot_positions: Foot end positions [6, 3] for stair detection
        """
        super().update(dt, velocity_command)
        
        # Stair detection if foot positions are provided
        if foot_positions is not None:
            self.detect_stair(foot_positions)
        
        # Smooth transition between flat and climbing parameters
        target_factor = 1.0 if self.is_stair_detected else 0.0
        self.transition_factor += np.clip(
            target_factor - self.transition_factor,
            -self.transition_rate * dt,
            self.transition_rate * dt
        )
        
        # Interpolate parameters
        self.params.step_height = self._lerp(
            self.default_params.step_height,
            self.climb_params.step_height,
            self.transition_factor
        )
        self.params.step_length = self._lerp(
            self.default_params.step_length,
            self.climb_params.step_length,
            self.transition_factor
        )
        self.params.cycle_time = self._lerp(
            self.default_params.cycle_time,
            self.climb_params.cycle_time,
            self.transition_factor
        )
    
    def get_leg_phase(self, leg_idx: int) -> float:
        """Use wave gait phase distribution"""
        offset = WaveGait.LEG_PHASE_OFFSETS[leg_idx]
        phase = ((self.time / self.params.cycle_time) + offset) % 1.0
        return phase
    
    def is_stance_phase(self, leg_idx: int) -> bool:
        """Check if in stance phase"""
        phase = self.get_leg_phase(leg_idx)
        return phase < self.params.duty_factor
    
    def get_foot_trajectory(self, leg_idx: int,
                            home_position: np.ndarray,
                            velocity_command: np.ndarray,
                            current_foot_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute foot trajectory with stair climbing support
        
        Stance phase: maintain current foot height (do not force to 0)
        Swing phase: front legs lift higher, consider y direction
        """
        phase = self.get_leg_phase(leg_idx)
        vy = float(velocity_command[1]) if len(velocity_command) > 1 else 0.0
        
        if self.is_stance_phase(leg_idx):
            # Stance phase - maintain foot height
            duty = max(self.params.duty_factor, 0.1)
            stance_progress = phase / duty
            stance_progress = np.clip(stance_progress, 0.0, 1.0)
            
            stride = self.params.step_length
            x_offset = stride * (0.5 - stance_progress)
            y_offset = vy * self.params.cycle_time * duty * (0.5 - stance_progress)
            
            target_pos = home_position.copy()
            target_pos[0] += x_offset
            target_pos[1] += y_offset
            # Maintain current foot height instead of forcing to 0
            if current_foot_pos is not None:
                target_pos[2] = current_foot_pos[2]
            else:
                target_pos[2] = home_position[2]
            
            vx_speed = -stride / (self.params.cycle_time * duty) if self.params.cycle_time * duty > 1e-6 else 0.0
            vy_speed = -vy if abs(vy) > 1e-6 else 0.0
            target_vel = np.array([vx_speed, vy_speed, 0])
            
        else:
            # Swing phase
            swing_denom = 1.0 - self.params.duty_factor
            swing_denom = max(swing_denom, 0.1)
            swing_progress = (phase - self.params.duty_factor) / swing_denom
            swing_progress = np.clip(swing_progress, 0.0, 1.0)
            
            stride = self.params.step_length
            x_offset = stride * (swing_progress - 0.5)
            y_offset = vy * self.params.cycle_time * (1 - self.params.duty_factor) * (swing_progress - 0.5)
            
            # Base leg lift height
            base_height = self.params.step_height * np.sin(swing_progress * np.pi)
            
            # Front legs get additional lift to ascend steps
            extra_lift = 0.0
            if leg_idx in self.FRONT_LEGS and self.transition_factor > 0.1:
                extra_lift = 0.03 * self.transition_factor * np.sin(swing_progress * np.pi)
            
            z_height = base_height + extra_lift
            
            target_pos = home_position.copy()
            target_pos[0] += x_offset
            target_pos[1] += y_offset
            target_pos[2] = home_position[2] - z_height  # Lift relative to home
            
            # Velocity
            swing_duration = self.params.cycle_time * (1 - self.params.duty_factor)
            if swing_duration > 1e-6:
                vx_speed = stride / swing_duration
                vz = (self.params.step_height * np.pi * np.cos(swing_progress * np.pi)) / swing_duration
            else:
                vx_speed = 0.0
                vz = 0.0
            vy_speed = vy if abs(vy) > 1e-6 else 0.0
            target_vel = np.array([vx_speed, vy_speed, -vz])
            
            # Limit velocity
            max_vel = 5.0
            vel_norm = np.linalg.norm(target_vel)
            if vel_norm > max_vel:
                target_vel = target_vel * (max_vel / vel_norm)
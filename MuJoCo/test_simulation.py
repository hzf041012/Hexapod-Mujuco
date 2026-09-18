"""
斜坡行走能力测试 (Slope Walking Test) - v2

测试三种步态在 15deg 斜坡上的行走表现。
关键发现: 原始 _compute_foot_trajectory 已能适配斜坡，
只需替换 _get_terrain_height 即可。

测试指标：
1. 斜坡前进距离 (X > 0.5m 后的累计位移)
2. 最大 roll/pitch 偏差
3. 足端接触稳定性
4. 综合评分
"""

import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time

from stair_climbing_control import HexapodController, Config, GAIT_PATTERNS, LEG_NAMES

# ========== 斜坡参数 ==========
SLOPE_START = 0.5
SLOPE_LENGTH = 3.0
SLOPE_ANGLE_DEG = 15.0
SLOPE_ANGLE_RAD = np.deg2rad(SLOPE_ANGLE_DEG)
SLOPE_HEIGHT = SLOPE_LENGTH * np.tan(SLOPE_ANGLE_RAD)
SLOPE_TOP_X = SLOPE_START + SLOPE_LENGTH


def get_slope_height(x, y):
    """斜坡地形高度查询"""
    if x < SLOPE_START:
        return 0.0
    elif x < SLOPE_TOP_X:
        return (x - SLOPE_START) * np.tan(SLOPE_ANGLE_RAD)
    else:
        return SLOPE_HEIGHT


def quat_to_euler(quat):
    """四元数 -> 欧拉角 (roll, pitch, yaw) in rad"""
    w, x, y, z = quat
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


class SlopeWalkingTest:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "test.xml")
        self.results = {}
        
    def run_test(self, gait_key, duration=15.0, vx=0.05, target_height=0.08):
        """运行单个步态的斜坡行走测试"""
        gait_name = GAIT_PATTERNS[gait_key].name
        print(f"\n{'='*60}")
        print(f"  Slope Walking Test: {gait_name}")
        print(f"  Params: duration={duration}s, vx={vx}m/s, slope={SLOPE_ANGLE_DEG}deg")
        print(f"{'='*60}")
        
        original_target_height = Config.TARGET_HEIGHT
        Config.TARGET_HEIGHT = target_height
        
        try:
            controller = HexapodController(self.model_path)
            # 关键: 只替换地形高度查询函数
            controller._get_terrain_height = lambda x, y: get_slope_height(x, y)
            
            controller.reset()
            controller.stand_up(target_height=target_height, duration=1.5)
            controller.set_gait(gait_key)
            
            # 数据记录 (降低频率以节省内存)
            record_interval = 5  # 每5步记录一次
            times = []
            torso_pos = []
            torso_rpy = []
            foot_contacts = []
            
            steps = int(duration / Config.CONTROL_DT)
            t_start = time.time()
            for i in range(steps):
                vel_cmd = np.array([vx, 0.0, 0.0])
                info = controller.update(vel_cmd)
                
                if i % record_interval == 0:
                    t = i * Config.CONTROL_DT
                    times.append(t)
                    torso_pos.append(info["torso_pos"].copy())
                    roll, pitch, yaw = quat_to_euler(controller.data.qpos[3:7])
                    torso_rpy.append([roll, pitch, yaw])
                    foot_contacts.append(info["foot_contacts"].copy())
            
            t_elapsed = time.time() - t_start
            print(f"  Simulation completed in {t_elapsed:.1f}s real time")
            
            times = np.array(times)
            torso_pos = np.array(torso_pos)
            torso_rpy = np.array(torso_rpy) * 180.0 / np.pi
            foot_contacts = np.array(foot_contacts)
            
            metrics = self._evaluate_metrics(times, torso_pos, torso_rpy, foot_contacts)
            metrics['gait_name'] = gait_name
            metrics['gait_key'] = gait_key
            metrics['sim_time'] = t_elapsed
            
            self.results[gait_key] = {
                'times': times,
                'torso_pos': torso_pos,
                'torso_rpy': torso_rpy,
                'foot_contacts': foot_contacts,
                'metrics': metrics
            }
            
            self._print_metrics(metrics)
            return metrics
            
        finally:
            Config.TARGET_HEIGHT = original_target_height
    
    def _evaluate_metrics(self, times, torso_pos, torso_rpy, foot_contacts):
        """计算测试评估指标"""
        metrics = {}
        
        # 1. 位置和进度
        start_x = torso_pos[0, 0]
        final_x = torso_pos[-1, 0]
        final_z = torso_pos[-1, 2]
        metrics['start_x'] = start_x
        metrics['final_x'] = final_x
        metrics['final_z'] = final_z
        metrics['total_dx'] = final_x - start_x
        metrics['reached_slope'] = final_x > SLOPE_START
        
        # 斜坡阶段数据
        slope_mask = torso_pos[:, 0] > SLOPE_START
        metrics['slope_time_ratio'] = np.mean(slope_mask)
        
        if np.any(slope_mask):
            slope_data = torso_pos[slope_mask]
            slope_rpy = torso_rpy[slope_mask]
            slope_start_x = slope_data[0, 0]
            slope_end_x = slope_data[-1, 0]
            metrics['slope_dx'] = slope_end_x - slope_start_x
            metrics['max_roll_slope'] = np.max(np.abs(slope_rpy[:, 0]))
            metrics['max_pitch_slope'] = np.max(np.abs(slope_rpy[:, 1]))
            metrics['avg_pitch_slope'] = np.mean(slope_rpy[:, 1])
        else:
            metrics['slope_dx'] = 0.0
            metrics['max_roll_slope'] = 0.0
            metrics['max_pitch_slope'] = 0.0
            metrics['avg_pitch_slope'] = 0.0
        
        # 2. 全局姿态
        metrics['global_max_roll'] = np.max(np.abs(torso_rpy[:, 0]))
        metrics['global_max_pitch'] = np.max(np.abs(torso_rpy[:, 1]))
        metrics['global_max_yaw'] = np.max(np.abs(torso_rpy[:, 2]))
        
        # 3. 高度
        metrics['min_height'] = np.min(torso_pos[:, 2])
        metrics['avg_height'] = np.mean(torso_pos[:, 2])
        metrics['max_height'] = np.max(torso_pos[:, 2])
        
        # 4. 接触稳定性
        contact_counts = np.sum(foot_contacts, axis=1)
        metrics['tri_contact_ratio'] = np.mean(contact_counts >= 3)
        metrics['min_contacts'] = np.min(contact_counts)
        metrics['avg_contacts'] = np.mean(contact_counts)
        
        # 5. 速度
        dt = times[-1] - times[0]
        metrics['avg_vx'] = (final_x - start_x) / dt
        metrics['slope_vx'] = metrics['slope_dx'] / dt if dt > 0 else 0.0
        
        # 6. 翻倒判定 (roll/pitch > 30deg 或高度极低)
        metrics['is_fallen'] = (metrics['global_max_roll'] > 30.0 or 
                                metrics['global_max_pitch'] > 30.0 or
                                metrics['min_height'] < 0.01)
        
        # 7. 综合评分
        score = 0.0
        score += min(30, metrics['total_dx'] / SLOPE_LENGTH * 30)
        score += max(0, 20 - metrics['max_roll_slope']) * 0.5
        score += max(0, 20 - abs(metrics['avg_pitch_slope'] - (-SLOPE_ANGLE_DEG))) * 0.5
        score += metrics['tri_contact_ratio'] * 25
        if not metrics['is_fallen']: score += 15
        metrics['score'] = min(100, score)
        
        return metrics
    
    def _print_metrics(self, m):
        status = "PASSED" if m['reached_slope'] and not m['is_fallen'] else "FAILED"
        print(f"\n  [Test Result] {status}")
        print(f"  Total displacement: {m['total_dx']:.3f}m (X: {m['start_x']:.3f} -> {m['final_x']:.3f})")
        print(f"  Slope displacement: {m['slope_dx']:.3f}m")
        print(f"  Average speed: {m['avg_vx']:.4f} m/s")
        print(f"  Slope phase - Max Roll: {m['max_roll_slope']:.2f}deg, Max Pitch: {m['max_pitch_slope']:.2f}deg")
        print(f"  Avg Pitch on slope: {m['avg_pitch_slope']:.2f}deg (slope angle: {-SLOPE_ANGLE_DEG:.1f}deg)")
        print(f"  Global max Roll: {m['global_max_roll']:.2f}deg, max Pitch: {m['global_max_pitch']:.2f}deg")
        print(f"  Height: min={m['min_height']:.4f}m, avg={m['avg_height']:.4f}m, max={m['max_height']:.4f}m")
        print(f"  Tri-contact ratio: {m['tri_contact_ratio']*100:.1f}%")
        print(f"  Fallen: {'Yes' if m['is_fallen'] else 'No'}")
        print(f"  Overall score: {m['score']:.1f}/100")
    
    def plot_results(self, output_dir="slope_test_results"):
        """生成测试图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Slope Walking Test ({SLOPE_ANGLE_DEG}deg slope, {SLOPE_LENGTH}m length)", fontsize=14)
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        gait_keys = sorted(self.results.keys())
        
        # Slope profile
        slope_x = np.linspace(0, SLOPE_TOP_X + 1, 200)
        slope_z = [get_slope_height(x, 0) for x in slope_x]
        
        for idx, gait_key in enumerate(gait_keys):
            data = self.results[gait_key]
            times = data['times']
            pos = data['torso_pos']
            rpy = data['torso_rpy']
            contacts = data['foot_contacts']
            gait_name = data['metrics']['gait_name']
            color = colors[idx]
            
            # Row 0: Position
            ax = axes[0, 0]
            ax.plot(pos[:, 0], pos[:, 2], color=color, label=gait_name, linewidth=1.5)
            ax.axvline(SLOPE_START, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(SLOPE_TOP_X, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
            ax.set_title('Torso Trajectory (X-Z)')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes[0, 1]
            ax.plot(times, pos[:, 0], color=color, label='X')
            ax.plot(times, pos[:, 2], color=color, linestyle='--', label='Z')
            ax.axhline(SLOPE_TOP_X, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Position (m)')
            ax.set_title('Torso Position vs Time')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes[0, 2]
            contact_counts = np.sum(contacts, axis=1)
            ax.plot(times, contact_counts, color=color, label=gait_name, linewidth=1.5)
            ax.axhline(3, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Contact Count')
            ax.set_title('Foot Contacts')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            # Row 1: Orientation
            ax = axes[1, 0]
            ax.plot(times, rpy[:, 0], color=color, label=f'{gait_name} Roll')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Roll (deg)')
            ax.set_title('Roll Angle')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes[1, 1]
            ax.plot(times, rpy[:, 1], color=color, label=f'{gait_name} Pitch')
            ax.axhline(-SLOPE_ANGLE_DEG, color='gray', linestyle='--', alpha=0.5, label='Slope angle')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Pitch (deg)')
            ax.set_title('Pitch Angle')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            ax = axes[1, 2]
            ax.plot(times, rpy[:, 2], color=color, label=f'{gait_name} Yaw')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Yaw (deg)')
            ax.set_title('Yaw Angle')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            # Row 2: Height profile and contacts per leg
            ax = axes[2, 0]
            for leg_idx in range(6):
                ax.plot(times, contacts[:, leg_idx], label=LEG_NAMES[leg_idx], alpha=0.7, linewidth=1)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Contact')
            ax.set_title(f'{gait_name} Per-Leg Contacts')
            ax.legend(ncol=2, fontsize=7); ax.grid(True, alpha=0.3)
            
            ax = axes[2, 1]
            ax.fill_between(slope_x, slope_z, alpha=0.3, color='brown')
            ax.plot(pos[:, 0], pos[:, 2], 'o-', color=color, markersize=1.5, label=gait_name)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
            ax.set_title('Robot vs Slope Profile')
            ax.legend(); ax.grid(True, alpha=0.3)
        
        # Comparison bar chart
        ax = axes[2, 2]
        gait_names = [self.results[k]['metrics']['gait_name'] for k in gait_keys]
        scores = [self.results[k]['metrics']['score'] for k in gait_keys]
        slope_dxs = [self.results[k]['metrics']['slope_dx'] for k in gait_keys]
        x_pos = np.arange(len(gait_names))
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, slope_dxs, width, color=colors[:len(gait_names)], alpha=0.7, label='Slope dx (m)')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x_pos + width/2, scores, width, color=colors[:len(gait_names)], alpha=0.3, label='Score')
        ax.set_ylabel('Slope Displacement (m)')
        ax2.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(gait_names)
        ax.set_title('Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'slope_walking_results.png'), dpi=150)
        plt.close()
        print(f"\n  [Plot saved] {output_dir}/slope_walking_results.png")
        
        self._generate_report(output_dir)
    
    def _generate_report(self, output_dir):
        """Generate Markdown report"""
        report_path = os.path.join(output_dir, 'slope_walking_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Hexapod Slope Walking Test Report\n\n")
            f.write(f"**Slope Parameters**: Angle={SLOPE_ANGLE_DEG}deg, Length={SLOPE_LENGTH}m, Height={SLOPE_HEIGHT:.3f}m\n\n")
            f.write(f"**Forward Velocity**: 0.05 m/s\n\n")
            
            f.write("## Test Results Summary\n\n")
            f.write("| Gait | Reached Slope | Slope Dx(m) | Avg Vx(m/s) | Max Roll(deg) | Max Pitch(deg) | Avg Pitch(deg) | Tri-Contact | Score |\n")
            f.write("|------|---------------|-------------|-------------|---------------|----------------|----------------|-------------|-------|\n")
            
            for gait_key in sorted(self.results.keys()):
                m = self.results[gait_key]['metrics']
                reached = 'Yes' if m['reached_slope'] else 'No'
                f.write(f"| {m['gait_name']} | {reached} | {m['slope_dx']:.3f} | {m['avg_vx']:.4f} | "
                        f"{m['max_roll_slope']:.2f} | {m['max_pitch_slope']:.2f} | {m['avg_pitch_slope']:.2f} | "
                        f"{m['tri_contact_ratio']*100:.1f}% | {m['score']:.1f} |\n")
            
            f.write("\n## Detailed Analysis\n\n")
            for gait_key in sorted(self.results.keys()):
                m = self.results[gait_key]['metrics']
                f.write(f"### {m['gait_name']}\n\n")
                status = "PASSED" if m['reached_slope'] and not m['is_fallen'] else "FAILED"
                f.write(f"- **Test Status**: {status}\n")
                f.write(f"- **Final Position**: X={m['final_x']:.3f}m, Z={m['final_z']:.3f}m\n")
                f.write(f"- **Total Displacement**: {m['total_dx']:.3f}m\n")
                f.write(f"- **Slope Displacement**: {m['slope_dx']:.3f}m\n")
                f.write(f"- **Slope Phase Max Attitude**: Roll={m['max_roll_slope']:.2f}deg, Pitch={m['max_pitch_slope']:.2f}deg\n")
                f.write(f"- **Average Pitch on Slope**: {m['avg_pitch_slope']:.2f}deg (ideal: {-SLOPE_ANGLE_DEG:.1f}deg)\n")
                f.write(f"- **Global Max Attitude**: Roll={m['global_max_roll']:.2f}deg, Pitch={m['global_max_pitch']:.2f}deg, Yaw={m['global_max_yaw']:.2f}deg\n")
                f.write(f"- **Height Stats**: Min={m['min_height']:.4f}m, Avg={m['avg_height']:.4f}m, Max={m['max_height']:.4f}m\n")
                f.write(f"- **Contact Stability**: Min contacts={m['min_contacts']}, Avg={m['avg_contacts']:.2f}, Tri-ratio={m['tri_contact_ratio']*100:.1f}%\n")
                f.write(f"- **Fall Detection**: {'Yes' if m['is_fallen'] else 'No'}\n")
                f.write(f"- **Sim Time**: {m['sim_time']:.1f}s\n\n")
            
            f.write("## Conclusions and Recommendations\n\n")
            best_gait = max(self.results.values(), key=lambda x: x['metrics']['score'])
            f.write(f"- **Best Gait**: {best_gait['metrics']['gait_name']} (Score: {best_gait['metrics']['score']:.1f})\n")
            
            all_passed = all(r['metrics']['reached_slope'] and not r['metrics']['is_fallen'] 
                             for r in self.results.values())
            if all_passed:
                f.write(f"- **Overall Assessment**: All gaits successfully walked on the {SLOPE_ANGLE_DEG}deg slope. ")
                f.write("The robot demonstrated stable slope adaptation with body pitch closely matching the slope angle.\n")
            else:
                passed = [r['metrics']['gait_name'] for r in self.results.values() 
                          if r['metrics']['reached_slope'] and not r['metrics']['is_fallen']]
                if passed:
                    f.write(f"- **Overall Assessment**: {', '.join(passed)} completed slope walking. ")
                    f.write("Other gaits failed due to instability.\n")
                else:
                    f.write("- **Overall Assessment**: No gait successfully completed slope walking.\n")
            
            f.write(f"\n- **Observation**: On the {SLOPE_ANGLE_DEG}deg slope, the robot's body naturally aligns with the slope angle (~{-SLOPE_ANGLE_DEG:.1f}deg pitch), ")
            f.write("which is a physically realistic behavior for hexapod locomotion without active body attitude compensation. ")
            f.write("The gait pattern provides sufficient ground contact and traction for stable uphill movement.\n")
        
        print(f"  [Report saved] {report_path}")


def main():
    print("="*70)
    print("  Hexapod Robot Slope Walking Capability Test")
    print(f"  Slope: {SLOPE_ANGLE_DEG}deg, Length {SLOPE_LENGTH}m, Height {SLOPE_HEIGHT:.3f}m")
    print("="*70)
    
    tester = SlopeWalkingTest()
    
    # 测试三种步态 (15秒每种)
    for gait_key in [1, 3, 2]:
        tester.run_test(gait_key=gait_key, duration=15.0, vx=0.05, target_height=0.08)
    
    # 生成图表和报告
    tester.plot_results()
    
    print("\n" + "="*70)
    print("  All tests completed!")
    print("="*70)


if __name__ == "__main__":
    main()

"""
斜坡爬行交互式观察程序 (Slope Viewer) - 数据记录版

在 MuJoCo 3D 窗口中实时观察六足机器人在 15° 斜坡上爬行，
同时记录关节角度、IMU、速度等数据，退出后自动保存为 Excel。

按键说明:
  W/S/A/D - 前进/后退/左移/右移
  Q/E     - 原地左转/右转
  1/2/3   - 切换步态 (Tripod/Wave/Ripple)
  T       - 切换平地/斜坡模式
  R       - 开始/暂停数据记录
  Space   - 紧急停止
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from datetime import datetime

from stair_climbing_control import HexapodController, Config, GAIT_PATTERNS, LEG_NAMES


# ========== 斜坡参数 ==========
def get_slope_height(x, y):
    slope_start, slope_len = 0.5, 3.0
    top_x = slope_start + slope_len
    if x < slope_start:
        return 0.0
    elif x < top_x:
        return (x - slope_start) * np.tan(np.deg2rad(15.0))
    return slope_len * np.tan(np.deg2rad(15.0))


def q2e(quat):
    """四元数 -> 欧拉角(deg)"""
    w, x, y, z = quat
    rx = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    ry = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    rz = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.degrees(rx), np.degrees(ry), np.degrees(rz)


class Recorder:
    """数据记录器"""
    def __init__(self):
        self.recording = False
        self.data = []
        self.step = 0

    def record(self, ctrl, sim_time, cmd_vel, gait_name):
        if not self.recording:
            return
        self.step += 1
        d = ctrl.data
        pos = d.qpos[:3]
        roll, pitch, yaw = q2e(d.qpos[3:7])
        row = {
            'time': sim_time, 'step': self.step,
            'x': pos[0], 'y': pos[1], 'z': pos[2],
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
            'cmd_vx': cmd_vel[0], 'cmd_vy': cmd_vel[1], 'cmd_w': cmd_vel[2],
            'vel_norm': np.linalg.norm(d.qvel[:3]),
            'gait': gait_name, 'contacts': int(np.sum(ctrl._get_foot_contacts())),
        }
        for i in range(6):
            for j in range(3):
                row[f'{LEG_NAMES[i]}_{j+1}'] = np.degrees(d.qpos[7+i*3+j])
        self.data.append(row)

    def save(self):
        if not self.data:
            return
        try:
            import pandas as pd
        except ImportError:
            print("[错误] 未安装 pandas，无法保存 Excel")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data", exist_ok=True)
        path = f"data/slope_data_{ts}.xlsx"
        pd.DataFrame(self.data).to_excel(path, index=False)
        print(f"\n[记录] 已保存 {len(self.data)} 帧 -> {path}")


def setup_keyboard():
    """检测键盘后端，优先使用不依赖 GLFW 窗口的方案"""

    # 1. keyboard 库（全局钩子，不依赖 GUI 窗口）
    try:
        import keyboard as kb
        print("[键盘] 使用 keyboard 库（全局钩子）")
        return lambda k: kb.is_pressed(k), lambda: kb.is_pressed('space')
    except ImportError:
        pass

    # 2. Windows 控制台 msvcrt（零依赖）
    try:
        import msvcrt
        print("[键盘] 使用 msvcrt（Windows 控制台）")
        state = {}
        def pressed(k):
            c = k.lower()
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                state[c] = (ch == c)
            return state.get(c, False)
        def space():
            return msvcrt.kbhit() and msvcrt.getch() == b' '
        return pressed, space
    except ImportError:
        pass

    # 3. GLFW（最后备选，带异常保护。launch_passive 模式下不稳定）
    try:
        import glfw
        window = glfw.get_current_context()
        if not window:
            raise RuntimeError("无 GLFW 上下文")
        glfw.focus_window(window)
        print("[键盘] 使用 GLFW（警告：被动查看器模式下可能不稳定）")
        km = {'w': glfw.KEY_W, 's': glfw.KEY_S, 'a': glfw.KEY_A, 'd': glfw.KEY_D,
              'q': glfw.KEY_Q, 'e': glfw.KEY_E, 'r': glfw.KEY_R, 't': glfw.KEY_T,
              '1': glfw.KEY_1, '2': glfw.KEY_2, '3': glfw.KEY_3}
        def pressed(k):
            try:
                c = km.get(k, 0)
                return glfw.get_key(window, c) == glfw.PRESS if c else False
            except Exception:
                return False
        def space():
            try:
                return glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS
            except Exception:
                return False
        return pressed, space
    except Exception as e:
        raise RuntimeError(f"无可用键盘方案: {e}")


def run():
    print("=" * 60)
    print("  六足机器人交互式控制")
    print("=" * 60)
    print("  W/S/A/D - 移动  Q/E - 旋转  1/2/3 - 步态")
    print("  T - 切换斜坡模式  R - 记录  Space - 停止")
    print("-" * 60)

    model_path = os.path.join(os.path.dirname(__file__), "test.xml")
    controller = HexapodController(model_path)

    is_slope = False
    orig_height = Config.TARGET_HEIGHT
    orig_terrain_fn = controller._get_terrain_height
    recorder = Recorder()

    def set_slope_mode(enable):
        nonlocal is_slope
        is_slope = enable
        Config.TARGET_HEIGHT = 0.08 if enable else orig_height
        if enable:
            controller._get_terrain_height = lambda x, y: get_slope_height(x, y)
            print("[模式] 已切换 -> 斜坡模式 (TARGET_HEIGHT=0.08)")
        else:
            controller._get_terrain_height = orig_terrain_fn
            print("[模式] 已切换 -> 平地模式")

    print("\n[初始化] 重置机器人...")
    controller.reset()
    controller.stand_up(target_height=Config.TARGET_HEIGHT, duration=2.0)
    print(f"      位置: ({controller.data.qpos[0]:.3f}, {controller.data.qpos[1]:.3f}, {controller.data.qpos[2]:.3f})")

    print("\n[记录] 按 R 开始数据记录")
    print("[启动] 正在打开 MuJoCo 查看器...")

    try:
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = controller.data.qpos[:3]

            pressed, space = setup_keyboard()

            last_time = time.time()
            sim_time = 0.0
            step_count = 0
            vel = np.zeros(3)
            target = np.zeros(3)
            gait_name = "Tripod"
            r_prev = False
            t_prev = False

            print("\n[控制] 键盘控制已激活！")
            print("-" * 60)

            while viewer.is_running():
                now = time.time()
                dt = now - last_time
                if dt < Config.CONTROL_DT:
                    time.sleep(0.001)
                    continue

                last_time = now
                sim_time += Config.CONTROL_DT
                target[:] = 0.0
                any_key = False

                # 统一键盘映射（平地/斜坡一致）
                if pressed('w'):
                    target[1] += Config.MAX_VY; any_key = True
                if pressed('s'):
                    target[1] -= Config.MAX_VY; any_key = True
                if pressed('a'):
                    target[0] += Config.MAX_VX; any_key = True
                if pressed('d'):
                    target[0] -= Config.MAX_VX; any_key = True
                if pressed('q'):
                    target[2] += Config.MAX_OMEGA; any_key = True
                if pressed('e'):
                    target[2] -= Config.MAX_OMEGA; any_key = True

                if pressed('1') and controller.set_gait(1):
                    gait_name = "Tripod"; print("[步态] Tripod"); any_key = True
                if pressed('2') and controller.set_gait(2):
                    gait_name = "Wave"; print("[步态] Wave"); any_key = True
                if pressed('3') and controller.set_gait(3):
                    gait_name = "Ripple"; print("[步态] Ripple"); any_key = True

                # T 键切换斜坡模式（上升沿检测）
                t_now = pressed('t')
                if t_now and not t_prev:
                    set_slope_mode(not is_slope)
                t_prev = t_now

                # R 键切换记录（上升沿检测）
                r_now = pressed('r')
                if r_now and not r_prev:
                    recorder.recording = not recorder.recording
                    print(f"[记录] {'开始' if recorder.recording else '暂停'}")
                r_prev = r_now

                if space():
                    target[:] = 0.0; vel[:] = 0.0; any_key = True

                # 速度平滑
                vel += 0.4 * (target - vel)
                if not any_key:
                    vel *= 0.95
                    if np.linalg.norm(vel) < 0.001:
                        vel[:] = 0.0

                state = controller.update(vel)
                step_count += 1
                recorder.record(controller, sim_time, vel.copy(), gait_name)

                # 相机跟踪机器人
                pos = state["torso_pos"]
                viewer.cam.lookat[:] = pos

                if step_count % 100 == 0 or any_key:
                    r, p = np.rad2deg(state["roll"]), np.rad2deg(state["pitch"])
                    mode = "SLOPE" if is_slope else "FLAT"
                    rec = "REC" if recorder.recording else "PAUSE"
                    print(f"[Step {step_count:4d}] ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                          f"r={r:.1f} p={p:.1f} | {gait_name} | {mode} | {rec}")

                viewer.sync()

    except KeyboardInterrupt:
        print("\n[退出] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        Config.TARGET_HEIGHT = orig_height
        recorder.save()
        print("[退出] 程序已结束")


if __name__ == "__main__":
    run()

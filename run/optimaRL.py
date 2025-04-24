import os
import csv
import numpy as np
import gymnasium as gym
from mujoco import (
    MjModel, MjData,
    mj_step, mj_resetData, mj_forward,
    mj_name2id, mjtObj
)
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class GraspMetricEnv(gym.Env):
    """
    RL environment for grasping with a shaped reward.
    По окончании каждого эпизода дописывает:
      total_reward, energy, overshoot, vel_pen, sym_pen
    в rl_metrics.csv
    """
    def __init__(self,
                 xml_path: str,
                 target_force: float = 1.0,
                 overshoot_thresh: float = 5.0,
                 grasp_success_thresh: float = 1.0,
                 max_steps: int = 100,
                 success_reward: float = 100.0):
        super().__init__()
        # MuJoCo model & data
        self.model = MjModel.from_xml_path(xml_path)
        self.data  = MjData(self.model)
        mj_forward(self.model, self.data)

        # Sensor IDs
        self.id_f_left  = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "left_contact")
        self.id_f_right = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "right_contact")
        self.id_p_left  = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "left_driver_pos")
        self.id_p_right = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "right_driver_pos")
        self.id_v_left  = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "left_driver_vel")
        self.id_v_right = mj_name2id(self.model, mjtObj.mjOBJ_SENSOR, "right_driver_vel")
        for name, sid in [
            ("left_contact", self.id_f_left),
            ("right_contact", self.id_f_right),
            ("left_driver_pos", self.id_p_left),
            ("right_driver_pos", self.id_p_right),
            ("left_driver_vel", self.id_v_left),
            ("right_driver_vel", self.id_v_right),
        ]:
            if sid < 0:
                raise RuntimeError(f"Sensor '{name}' not found in model")

        # Parameters
        self.target_force         = target_force
        self.overshoot_thresh     = overshoot_thresh
        self.grasp_success_thresh = grasp_success_thresh
        self.max_steps            = max_steps
        self.success_reward       = success_reward
        self.action_scale         = 20.0   # масштаб управления

        # Action & observation spaces
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        obs_dim = 6 + 4
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self.data.ctrl[0] = 0.0
        mj_forward(self.model, self.data)
        # Episode accumulators
        self.step_count     = 0
        self.energy_acc     = 0.0
        self.over_acc       = 0.0
        self.vel_acc        = 0.0
        self.sym_acc        = 0.0
        self.episode_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        lf = self.data.sensordata[self.id_f_left*3 : self.id_f_left*3+3]
        rf = self.data.sensordata[self.id_f_right*3: self.id_f_right*3+3]
        p_l = self.data.sensordata[self.id_p_left]
        p_r = self.data.sensordata[self.id_p_right]
        v_l = self.data.sensordata[self.id_v_left]
        v_r = self.data.sensordata[self.id_v_right]
        return np.concatenate([lf, rf, [p_l, p_r, v_l, v_r]]).astype(np.float32)

    def step(self, action):
        dt = self.model.opt.timestep
        # Base step penalty
        reward = -1.0

        # Apply action
        delta = float(action[0]) * self.action_scale
        self.data.ctrl[0] = np.clip(self.data.ctrl[0] + delta, 0.0, 255.0)
        mj_step(self.model, self.data)
        obs = self._get_obs()

        # Compute forces
        F_L, F_R = np.linalg.norm(obs[0:3]), np.linalg.norm(obs[3:6])
        Ftot = F_L + F_R

        # small positive reward for force
        reward += 0.1 * (Ftot / self.grasp_success_thresh)

        # Penalties
        e      = abs(self.data.ctrl[0]) * dt
        over   = max(Ftot - self.overshoot_thresh, 0.0) * dt
        vpen   = (abs(obs[8]) + abs(obs[9])) * dt
        sympen = abs(obs[6] - obs[7]) * dt
        reward -= 0.05 * e + 10.0 * over + 0.1 * vpen + 0.1 * sympen

        # Accumulate for logging
        self.energy_acc += e
        self.over_acc   += over
        self.vel_acc    += vpen
        self.sym_acc    += sympen
        self.episode_reward += reward

        # Success
        done = False
        if Ftot >= self.grasp_success_thresh:
            reward += self.success_reward
            self.episode_reward += self.success_reward
            done = True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        # On episode end, append metrics to CSV
        if done:
            # ensure header
            if not os.path.exists("rl_metrics.csv"):
                with open("rl_metrics.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "total_reward", "energy",
                        "overshoot", "vel_pen", "sym_pen"
                    ])
            # append values
            with open("rl_metrics.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{self.episode_reward:.4f}",
                    f"{self.energy_acc:.4f}",
                    f"{self.over_acc:.4f}",
                    f"{self.vel_acc:.4f}",
                    f"{self.sym_acc:.4f}",
                ])

            print(
                f"[End] total_reward={self.episode_reward:.2f}, "
                f"energy={self.energy_acc:.1f}, "
                f"overshoot={self.over_acc:.3f}, "
                f"vel_pen={self.vel_acc:.3f}, "
                f"sym_pen={self.sym_acc:.3f}"
            )

        return obs, reward, done, False, {}

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, print_freq: int = 5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq      = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            rem = self.total_timesteps - self.num_timesteps
            print(f"[Progress] {self.num_timesteps}/{self.total_timesteps}, remaining {rem}")
        return True

if __name__ == "__main__":
    # Перед обучением удалим старый лог
    if os.path.exists("rl_metrics.csv"):
        os.remove("rl_metrics.csv")

    raw_env = GraspMetricEnv("doms_pr/model/scene.xml")
    env     = Monitor(raw_env, filename="metric_monitor.csv")

    total_steps = 50_000
    model       = SAC("MlpPolicy", env, verbose=0)
    progress_cb = ProgressCallback(total_timesteps=total_steps, print_freq=10_000)

    model.learn(
        total_timesteps=total_steps,
        log_interval=None,
        callback=progress_cb
    )
    model.save("grasp_metric_sac")

    # После окончания в rl_metrics.csv будут все эпизодные метрики,
    # а в metric_monitor.csv — стандартные r и l.

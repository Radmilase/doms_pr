import mujoco
import numpy as np
from scipy.optimize import minimize, dual_annealing, differential_evolution
import time
import matplotlib.pyplot as plt
import pandas as pd


model = mujoco.MjModel.from_xml_path("doms_pr\\model\\scene.xml")
data = mujoco.MjData(model)

print("actuator name at ctrl[0]:", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0))

sensor_ids = {
    'force': {
        'left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact"),
        'right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")
    },
    'position': {
        'left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_driver_pos"),
        'right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_driver_pos")
    },
    'velocity': {
        'left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_driver_vel"),
        'right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_driver_vel")
    }
}

for sensor_type in sensor_ids:
    for side in sensor_ids[sensor_type]:
        if sensor_ids[sensor_type][side] == -1:
            raise ValueError(f"Сенсор {sensor_type}/{side} не найден")

def simulate(Kp, Kd, target_force):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    data.ctrl[0] = 50.0

    for _ in range(10):
        mujoco.mj_step(model, data)

    total_force_error = 0
    energy = 0
    overshoot_penalty = 0
    velocity_penalty = 0
    symmetry_penalty = 0
    previous_error = 0
    contact_frames = 0
    grasp_success = 0
    total_steps = 300

    for step in range(total_steps):
        mujoco.mj_step(model, data)

        start = model.sensor_adr[sensor_ids['force']['left']]
        dim = model.sensor_dim[sensor_ids['force']['left']]
        left_force = data.sensordata[start:start+dim]
        start = model.sensor_adr[sensor_ids['force']['right']]
        dim = model.sensor_dim[sensor_ids['force']['right']]
        right_force = data.sensordata[start:start+dim]
        total_force = np.nan_to_num(np.linalg.norm(left_force)) + np.nan_to_num(np.linalg.norm(right_force))

        start = model.sensor_adr[sensor_ids['position']['left']]
        left_pos = data.sensordata[start]
        start = model.sensor_adr[sensor_ids['position']['right']]
        right_pos = data.sensordata[start]
        start = model.sensor_adr[sensor_ids['velocity']['left']]
        left_vel = data.sensordata[start]
        start = model.sensor_adr[sensor_ids['velocity']['right']]
        right_vel = data.sensordata[start]

        error = target_force - total_force
        derivative = (error - previous_error)/model.opt.timestep
        control_signal = Kp * error + Kd * derivative
        previous_error = error

        data.ctrl[0] = np.clip(data.ctrl[0] + control_signal, 0, 255)

        total_force_error += error**2
        energy += abs(data.ctrl[0])
        if total_force > 5.0:
            overshoot_penalty += (total_force - 5.0)**2

        if total_force > 0.5:
            contact_frames += 1
        if total_force > 1.5:
            grasp_success += 1

        velocity_penalty += abs(left_vel) + abs(right_vel)
        symmetry_penalty += abs(left_pos - right_pos)

    grasp_ratio = grasp_success / total_steps

    if contact_frames < 1:
        return 9999

    return -grasp_ratio + 0.05 * energy + 10 * overshoot_penalty + 0.1 * velocity_penalty + 0.1 * symmetry_penalty

def objective(params):
    Kp, Kd, target_force = params
    score = simulate(Kp, Kd, target_force)
    print(f"Проверка: Kp={Kp:.4f}, Kd={Kd:.4f}, Target={target_force:.4f} -> Score={score:.4f}")
    return score

def run_brute_force():
    Kp_values = np.linspace(0.01, 0.2, 4)
    Kd_values = np.linspace(0.001, 0.02, 4)
    Target_values = np.linspace(0.5, 2.0, 4)
    best_score = float('inf')
    best_params = None
    history = []
    for Kp in Kp_values:
        for Kd in Kd_values:
            for T in Target_values:
                score = objective([Kp, Kd, T])
                history.append((Kp, Kd, T, score))
                if score < best_score:
                    best_score = score
                    best_params = (Kp, Kd, T)
    return best_params, best_score, history


def run_annealing():
    bounds = [(0.01, 1.0), (0.001, 0.1), (0.5, 3.0)]
    res = dual_annealing(objective, bounds)
    return res.x, res.fun, [(res.x[0], res.x[1], res.x[2], res.fun)]

def run_genetic():
    bounds = [(0.01, 1.0), (0.001, 0.1), (0.5, 3.0)]
    res = differential_evolution(objective, bounds)
    return res.x, res.fun, [(res.x[0], res.x[1], res.x[2], res.fun)]

def plot_results(histories):
    labels = list(histories.keys())
    final_scores = [min([h[3] for h in histories[m]]) for m in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, final_scores)
    plt.ylabel("Минимальная целевая функция")
    plt.title("Сравнение методов оптимизации")
    plt.grid(True)
    plt.show()

def print_results_table(histories):
    print("\nСравнение результатов")
    header = "{:<20} | {:<15} | {:<35}".format("Метод", "Лучший score", "Параметры (Kp, Kd, Target)")
    print(header)
    print("-" * len(header))
    for method, trials in histories.items():
        best = min(trials, key=lambda x: x[3])
        Kp, Kd, T, score = best
        print(f"{method:<20} | {score:<15.4f} | ({Kp:.4f}, {Kd:.4f}, {T:.4f})")

if __name__ == '__main__':
    histories = {}
    print("\nBrute Force")
    p, s, h = run_brute_force()
    histories["Brute Force"] = h

    print("\nSimulated Annealing")
    p, s, h = run_annealing()
    histories["Annealing"] = h

    print("\nGenetic Algorithm")
    p, s, h = run_genetic()
    histories["Genetic"] = h

    print_results_table(histories)

    # ——— RL comparison ———

    # Путь к файлу с RL-метриками (его создавали в вашем RL-скрипте)
    rl_df = pd.read_csv("doms_pr\\run\\rl_metrics.csv")

    mean_rl = rl_df["total_reward"].mean()
    std_rl  = rl_df["total_reward"].std()

    print("\nRL (SAC) metrics:")
    print(f"  Mean Reward = {mean_rl:.4f}")
    print(f"  Std Dev     = {std_rl:.4f}")
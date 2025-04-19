import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Загрузка модели
m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

# Получение ID сенсоров
sensor_ids = {
    'force': {
        'left': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact"),
        'right': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")
    },
    'position': {
        'left': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_driver_pos"),
        'right': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_driver_pos")
    },
    'velocity': {
        'left': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_driver_vel"),
        'right': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_driver_vel")
    }
}

# Проверка сенсоров
for sensor_type in sensor_ids:
    for side in sensor_ids[sensor_type]:
        if sensor_ids[sensor_type][side] == -1:
            raise ValueError(f"Сенсор {sensor_type}/{side} не найден!")

# Параметры PD-регулятора
Kp = 0.1  
Kd = 0.01  
target_force = 1.0  
max_force = 5.0
previous_error = 0.0

def pd_control(current_force, target_force, dt):
    global previous_error
    error = target_force - current_force
    derivative = (error - previous_error)/dt if dt > 0 else 0
    output = Kp*error + Kd*derivative
    previous_error = error
    return output

def loosen_grip(control_signal):
    return control_signal - 1.0

# Структуры для хранения данных
data = {
    'time': [],
    'control': [],
    'force': {
        'left': [[], [], []],
        'right': [[], [], []],
        'total': []
    },
    'position': {
        'left': [],
        'right': []
    },
    'velocity': {
        'left': [],
        'right': []
    }
}

# Инициализация CSV
with open('sensor_data.csv', 'w', newline='') as f:
    headers = ['Timestamp', 'Control']
    headers += [f'Force_{side}_{comp}' for side in ['left', 'right'] for comp in ['X','Y','Z']]
    headers += [f'Pos_{side}' for side in ['left', 'right']]
    headers += [f'Vel_{side}' for side in ['left', 'right']]
    csv.writer(f).writerow(headers)

# Запуск симуляции
viewer = mujoco.viewer.launch_passive(m, d)

try:
    start_time = time.time()
    with open('sensor_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        
        while viewer.is_running() and time.time() - start_time < 20:
            sim_start = time.time()
            mujoco.mj_step(m, d)
            
            # Считывание данных сенсоров
            left_force = d.sensordata[sensor_ids['force']['left']*3 : sensor_ids['force']['left']*3+3]
            right_force = d.sensordata[sensor_ids['force']['right']*3 : sensor_ids['force']['right']*3+3]
            left_pos = d.sensordata[sensor_ids['position']['left']]
            right_pos = d.sensordata[sensor_ids['position']['right']]
            left_vel = d.sensordata[sensor_ids['velocity']['left']]
            right_vel = d.sensordata[sensor_ids['velocity']['right']]
            
            # Расчет суммарной силы
            total_force = np.nan_to_num(np.linalg.norm(left_force)) + np.nan_to_num(np.linalg.norm(right_force))
            
            # Управление
            control_signal = pd_control(total_force, target_force, m.opt.timestep)
            if total_force > max_force:
                control_signal = loosen_grip(control_signal)
            d.ctrl[0] = np.clip(d.ctrl[0] + control_signal, 0, 255)
            
            # Сохранение данных
            current_time = time.time() - start_time
            data['time'].append(current_time)
            data['control'].append(control_signal)
            data['force']['total'].append(total_force)
            
            for i in range(3):
                data['force']['left'][i].append(left_force[i])
                data['force']['right'][i].append(right_force[i])
            
            data['position']['left'].append(left_pos)
            data['position']['right'].append(right_pos)
            data['velocity']['left'].append(left_vel)
            data['velocity']['right'].append(right_vel)
            
            row = [current_time, control_signal]
            row += list(left_force) + list(right_force)
            row += [left_pos, right_pos, left_vel, right_vel]
            writer.writerow(row)
            
            # Синхронизация
            time_until_next_step = m.opt.timestep - (time.time() - sim_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            viewer.sync()

finally:
    viewer.close()
    print("Данные сохранены в sensor_data.csv")

    # Создание комплексного отчета
    fig = plt.figure(figsize=(15, 25))
    gs = GridSpec(7, 2, figure=fig)

    # 1. Силы и управление
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data['time'], data['force']['total'], 'b-', label='Суммарная сила')
    ax1.plot(data['time'], data['control'], 'r--', label='Управление')
    ax1.set_title('Динамика системы')
    ax1.legend()
    ax1.grid(True)

    # 2. Компоненты сил (исправленный блок)
    components = ['X', 'Y', 'Z']
    for i, comp in enumerate(components):
        # Используем разные строки для каждого компонента
        ax = fig.add_subplot(gs[i+1, :])  # Занимаем всю ширину строки
        ax.plot(data['time'], data['force']['left'][i], 'b-', label='Left')
        ax.plot(data['time'], data['force']['right'][i], 'r--', label='Right')
        ax.set_title(f'Компонента {comp} силы')
        ax.legend()
        ax.grid(True)

    # 3. Позиции (смещаем индексы на +3 после компонентов силы)
    ax3 = fig.add_subplot(gs[4, :])  # Было 2, стало 4 (0-3 заняты)
    ax3.plot(data['time'], data['position']['left'], 'b-', label='Left')
    ax3.plot(data['time'], data['position']['right'], 'r--', label='Right')
    ax3.set_title('Позиции драйверов')
    ax3.legend()
    ax3.grid(True)

    # 4. Скорости
    ax4 = fig.add_subplot(gs[5, :])  # Смещаем на +1
    ax4.plot(data['time'], data['velocity']['left'], 'b-', label='Left')
    ax4.plot(data['time'], data['velocity']['right'], 'r--', label='Right')
    ax4.set_title('Скорости драйверов')
    ax4.legend()
    ax4.grid(True)

    # 5. Сравнение позиций и скоростей
    ax5 = fig.add_subplot(gs[6, 0])
    ax5.plot(data['position']['left'], data['velocity']['left'], 'b-')
    ax5.set_title('Фазовый портрет (левый)')
    ax5.grid(True)

    ax6 = fig.add_subplot(gs[6, 1])
    ax6.plot(data['position']['right'], data['velocity']['right'], 'r-')
    ax6.set_title('Фазовый портрет (правый)')
    ax6.grid(True)

    plt.tight_layout()
    plt.show()
    # 6. Гистограммы
    ax7 = fig.add_subplot(gs[5, 0])
    ax7.hist(data['position']['left'], bins=50, alpha=0.7, label='Left')
    ax7.hist(data['position']['right'], bins=50, alpha=0.7, label='Right')
    ax7.set_title('Распределение позиций')
    ax7.legend()
    ax7.grid(True)

    ax8 = fig.add_subplot(gs[5, 1])
    ax8.hist(data['velocity']['left'], bins=50, alpha=0.7, label='Left')
    ax8.hist(data['velocity']['right'], bins=50, alpha=0.7, label='Right')
    ax8.set_title('Распределение скоростей')
    ax8.legend()
    ax8.grid(True)

    # 7. Корреляции
    ax9 = fig.add_subplot(gs[6, 0])
    ax9.scatter(data['position']['left'], data['velocity']['left'], s=1)
    ax9.set_title('Корреляция позиция-скорость (левый)')
    ax9.grid(True)

    ax10 = fig.add_subplot(gs[6, 1])
    ax10.scatter(data['position']['right'], data['velocity']['right'], s=1)
    ax10.set_title('Корреляция позиция-скорость (правый)')
    ax10.grid(True)

    plt.tight_layout()
    plt.show()

    # Дополнительные графики
    plt.figure(figsize=(12, 8))
    plt.suptitle('Сравнение всех параметров')

    for i, (param, values) in enumerate([('Позиция', data['position']), 
                                       ('Скорость', data['velocity'])]):
        plt.subplot(2, 1, i+1)
        plt.plot(data['time'], values['left'], 'b-', label='Left')
        plt.plot(data['time'], values['right'], 'r--', label='Right')
        plt.title(param)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

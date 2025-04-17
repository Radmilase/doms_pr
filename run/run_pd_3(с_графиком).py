import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

# Загрузка модели
m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

# ID сенсоров и актуаторов
left_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact")
right_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")
actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

# Проверка сенсоров
if left_sensor_id == -1 or right_sensor_id == -1:
    raise ValueError("Сенсоры не найдены!")

# Параметры PD-регулятора
Kp = 0.1  
Kd = 0.01  
target_force = 1.0  
max_force = 5.0
previous_error = 0.0

def pd_control(current_force, target_force, dt):
    global previous_error
    error = target_force - current_force
    derivative = (error - previous_error)/dt
    output = Kp*error + Kd*derivative
    previous_error = error
    return output

def loosen_grip(control_signal):
    control_signal -= 1.0
    return control_signal

# Списки для хранения данных
time_data = []
force_data = []
control_data = []
left_components = [[], [], []]
right_components = [[], [], []]

# CSV файл
with open('sensor_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Left_X', 'Left_Y', 'Left_Z', 
                    'Right_X', 'Right_Y', 'Right_Z', 'Control'])

# Запуск симуляции
viewer = mujoco.viewer.launch_passive(m, d)

try:
    start_time = time.time()
    with open('sensor_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        
        while viewer.is_running() and time.time() - start_time < 20:
            sim_start = time.time()
            mujoco.mj_step(m, d)
            
            # Чтение данных сенсоров
            left_force = d.sensordata[left_sensor_id*3 : left_sensor_id*3+3]
            right_force = d.sensordata[right_sensor_id*3 : right_sensor_id*3+3]
            
            # Расчет суммарной силы
            total_force = np.linalg.norm(left_force) + np.linalg.norm(right_force)
            
            # Управление
            control_signal = pd_control(total_force, target_force, m.opt.timestep)
            if total_force > max_force:
                control_signal = loosen_grip(control_signal)
            d.ctrl[0] = np.clip(d.ctrl[0] + control_signal, 0, 255)
            
            # Сохранение данных
            current_time = time.time() - start_time
            time_data.append(current_time)
            force_data.append(total_force)
            control_data.append(control_signal)
            
            for i in range(3):
                left_components[i].append(left_force[i])
                right_components[i].append(right_force[i])
            
            writer.writerow([current_time, *left_force, *right_force, control_signal])
            
            # Синхронизация
            time_until_next_step = sim_start + m.opt.timestep - time.time()
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            viewer.sync()

finally:
    viewer.close()
    print("Данные сохранены в sensor_data.csv")

    # Создаем фигуру с 5 графиками
    fig, axs = plt.subplots(5, 1, figsize=(12, 20))
    
    # 1. Суммарная сила и управляющий сигнал
    axs[0].plot(time_data, force_data, 'b-', label='Суммарная сила')
    axs[0].plot(time_data, control_data, 'r--', label='Управляющий сигнал')
    axs[0].set_title('Динамика захвата')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Компоненты левого сенсора
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[1].plot(time_data, left_components[i], 
                   color=colors[i], 
                   label=f'Left {labels[i]}')
    axs[1].set_title('Компоненты силы (левый сенсор)')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Компоненты правого сенсора
    for i in range(3):
        axs[2].plot(time_data, right_components[i],
                   color=colors[i],
                   label=f'Right {labels[i]}')
    axs[2].set_title('Компоненты силы (правый сенсор)')
    axs[2].legend()
    axs[2].grid(True)

    # 4. Сравнение X-компонент
    axs[3].plot(time_data, left_components[0], 'b-', label='Left X')
    axs[3].plot(time_data, right_components[0], 'r--', label='Right X')
    axs[3].set_title('Сравнение X-компонент')
    axs[3].legend()
    axs[3].grid(True)

    # 5. Сравнение Y-компонент
    axs[4].plot(time_data, left_components[1], 'b-', label='Left Y')
    axs[4].plot(time_data, right_components[1], 'r--', label='Right Y')
    axs[4].set_title('Сравнение Y-компонент')
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()

    # Дополнительный график Z-компонент
    plt.figure(figsize=(12,4))
    plt.plot(time_data, left_components[2], 'b-', label='Left Z')
    plt.plot(time_data, right_components[2], 'r--', label='Right Z')
    plt.title('Сравнение Z-компонент')
    plt.legend()
    plt.grid(True)
    plt.show()
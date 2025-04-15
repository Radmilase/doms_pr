import mujoco
import mujoco.viewer
import numpy as np
import time

# Загрузка модели из XML-файла
m = mujoco.MjModel.from_xml_path("D:/Itmo/DOMS/project/doms_project/model/scene.xml")
d = mujoco.MjData(m)

# Получаем ID сенсоров по именам (используем mj_name2id)
left_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact")
right_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")

# Проверяем, что сенсоры найдены
if left_sensor_id == -1 or right_sensor_id == -1:
    raise ValueError("Не удалось найти один из сенсоров!")

viewer = mujoco.viewer.launch_passive(m, d)  # Исправлено: mujoco.viewer, а не mujoco_viewer

# Устанавливаем частоту обновления (например, 60 Гц)
sim_dt = m.opt.timestep  # Шаг симуляции из модели
render_dt = 1.0 / 60.0  # Желаемая частота рендеринга (60 FPS)

try:
    while viewer.is_running:  # Главный цикл (работает, пока открыто окно)
        sim_start = time.time()
        
        # Шаг симуляции
        mujoco.mj_step(m, d)
        
        # Чтение данных с сенсоров
        left_force = d.sensordata[left_sensor_id*3 : left_sensor_id*3+3]
        right_force = d.sensordata[right_sensor_id*3 : right_sensor_id*3+3]
        
        print(f"Left force: {left_force}, Right force: {right_force}")
        
        # Синхронизация визуализации с реальным временем
        time_until_next_step = sim_start + sim_dt - time.time()
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
        viewer.sync()  # Обновляем визуализацию

finally:
    viewer.close()  # Корректное закрытие
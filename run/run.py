import mujoco
import mujoco.viewer
import numpy as np

# Загрузка модели из XML-файла
m = mujoco.MjModel.from_xml_path("D:/Itmo/DOMS/project/doms_project/model/2f85.xml")
d = mujoco.MjData(m)

# Получаем ID сенсоров по именам (используем mj_name2id)
left_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact")
right_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")

# Проверяем, что сенсоры найдены
if left_sensor_id == -1 or right_sensor_id == -1:
    raise ValueError("Не удалось найти один из сенсоров!")

viewer = mujoco.viewer.launch_passive(m, d)  # Исправлено: mujoco.viewer, а не mujoco_viewer

# Функция для шага симуляции и чтения сил
def step_and_read_forces(d):
    mujoco.mj_step(m, d)  # Лучше использовать mj_step для явного управления
    sensor_data = d.sensordata
    left_force = sensor_data[left_sensor_id*3 : left_sensor_id*3 + 3]  # force - вектор из 3 компонентов
    right_force = sensor_data[right_sensor_id*3 : right_sensor_id*3 + 3]
    return left_force, right_force

# Пример цикла симуляции
for i in range(1000):
    left_f, right_f = step_and_read_forces(d)
    print(f"Step {i}: Left force = {left_f}, Right force = {right_f}")
    viewer.sync()  # Обновляем визуализацию



# Загрузка модели
m = mujoco.MjModel.from_xml_path("D:/Itmo/DOMS/project/doms_project/model/scene.xml")
d = mujoco.MjData(m)

# ID сенсоров
left_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact")
right_sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")

# ID актуатора
actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

# Проверка сенсоров
if left_sensor_id == -1 or right_sensor_id == -1:
    raise ValueError("Сенсоры не найдены!")

# Параметры PID-регулятора
Kp = 1.0  # Пропорциональный коэффициент
Ki = 0.0  # Интегральный коэффициент
Kd = 0.1  # Дифференциальный коэффициент

# Целевое значение силы (когда объект захвачен)
target_force = 1.0  # Примерное значение, настройте под свою задачу

# Предыдущая ошибка для дифференциальной составляющей
previous_error = 0.0
integral = 0.0

# Функция PID-регулятора
def pid_control(current_force, target_force, dt):
    global previous_error, integral
    error = target_force - current_force
    integral = integral + error * dt
    derivative = (error - previous_error) / dt
    output = Kp * error + Ki * integral + Kd * derivative
    previous_error = error
    return output

# Запись данных в CSV
with open('sensor_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Left_Force_X', 'Left_Force_Y', 'Left_Force_Z',
                     'Right_Force_X', 'Right_Force_Y', 'Right_Force_Z', 'Control_Signal'])

# Запуск симуляции
viewer = mujoco.viewer.launch_passive(m, d)

try:
    start_time = time.time()
    with open('sensor_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        
        while viewer.is_running() and time.time() - start_time < 20:
            sim_start = time.time()
            
            # Шаг симуляции
            mujoco.mj_step(m, d)
            
            # Чтение данных с сенсоров
            left_force = d.sensordata[left_sensor_id*3 : left_sensor_id*3+3]
            right_force = d.sensordata[right_sensor_id*3 : right_sensor_id*3+3]
            
            # Вычисление суммарной силы (модуль вектора)
            total_force = np.linalg.norm(left_force) + np.linalg.norm(right_force)
            
            # PID-регулирование
            control_signal = pid_control(total_force, target_force, m.opt.timestep)
            
            # Применение управляющего сигнала к актуатору
            d.ctrl[0] = d.ctrl[0] + control_signal #Управление по усилию
            d.ctrl[0] = np.clip(d.ctrl[0], 0, 255)  # Ограничение управляющего сигнала
            
            # Запись данных в CSV
            writer.writerow([
                time.time() - start_time,
                *left_force,
                *right_force,
                control_signal
            ])
            
            # Синхронизация времени
            time_until_next_step = sim_start + m.opt.timestep - time.time()
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
            viewer.sync()

finally:
    viewer.close()
    print("Симуляция завершена, данные сохранены в sensor_data.csv")
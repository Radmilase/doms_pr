import time
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from  optimaRL import GraspMetricEnv

def visualize_rl_agent(
    model_path: str,
    xml_path: str,
    pause: float = None
):
    env   = GraspMetricEnv(xml_path)
    agent = SAC.load(model_path, env=env)

    obs, _ = env.reset()
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    done = False
    try:
        while viewer.is_running():
            if not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
            viewer.sync()
            if pause:
                time.sleep(pause)

    finally:
        # чтобы не вызывать viewer.close() до ручного закрытия
        pass

if __name__ == "__main__":
    visualize_rl_agent(
        model_path="doms_pr/run/grasp_metric_sac.zip",
        xml_path="doms_pr/model/scene3.xml",
        pause=0.05                             # 50 мс задержки между кадрами
    )

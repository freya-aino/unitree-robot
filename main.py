from pathlib import Path
from brax.io import mjcf
from brax import envs
# from brax.envs.wrappers import torch as torch_wrapper
# from brax.envs.wrappers import gym as gym_wrapper


# from unitree_robot.common.cdds import get_all_cdds_topics
# from unitree_robot.common.datastructure import NETWORK_INTERFACE
# from unitree_robot.train.environments import Go2BraxEnv, GymGo2Env


if __name__ == "__main__":

    MJCF_PATH = "./external/mjcf-robot-files/go2/scene.xml"
    # MJCF_ASSET_PATH = "./external/mjcf-robot-files/go2/assets/"

    # -- brax
    # sys = mjcf.load(MJCF_PATH)
    # env = Go2BraxEnv(sys)
    # print(type(env))

    # -- mjx
    # from mujoco import MjModel, MjData, mjx
    # mj_model = MjModel.from_xml_path(MJCF_PATH)
    # mj_data = MjData(mj_model)
    # mjx_model = mjx.put_model(mj_model)
    # mjx_data = mjx.put_data(mj_model, mj_data)
    # print(mjx_model.actuator_acc0)

    # -- gym
    # gym_env = GymGo2Env(MJCF_PATH) # same problem as the MjModel.from_xml_path directly
    # print(type(gym_env))


    from unitree_robot.train.environments import MujocoEnv
    from gymnasium.spaces import Space
    import numpy as np
    env = MujocoEnv(
        MJCF_PATH,
        sim_frames_per_step=5,
        observation_space=Space(
            shape=[24], # TODO: change
            dtype=np.float32,
        ),
    )

    # print(mj_model.actuator_ctrlrange)
    # print(mj_data.qpos.ravel())
    # print(mj_data.qvel.ravel())

    import time

    render_fps = 60

    # print(env.action_space)
    env.reset()
    

    try:
        for i in range(1000):

            # print(env.get_state_vector())
            t_start = time.perf_counter()

            action = np.random.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
            env.do_simulation(
                ctrl=action,
                n_frames=5
            )
            
            env.render()

            elapsed = time.perf_counter() - t_start
            target = 1.0 / render_fps
            if elapsed < target:
                time.sleep(target - elapsed)

            print(f"effective fps: {1.0/(time.perf_counter() - t_start)}")

    except Exception as e:
        raise e
    finally:

        env.close()
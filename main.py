from pathlib import Path
from brax.io import mjcf
from brax import envs
# from brax.envs.wrappers import torch as torch_wrapper
# from brax.envs.wrappers import gym as gym_wrapper


# from unitree_robot.common.cdds import get_all_cdds_topics
# from unitree_robot.common.datastructure import NETWORK_INTERFACE
from unitree_robot.train.environments import Go2BraxEnv, GymGo2Env


if __name__ == "__main__":

    MJCF_PATH = "./external/mjcf-robot-files/go2/scene.xml"
    # MJCF_ASSET_PATH = "./external/mjcf-robot-files/go2/assets/"

    # -- brax
    # sys = mjcf.load(MJCF_PATH)
    # env = Go2BraxEnv(sys)
    # print(type(env))

    # -- mjx
    from mujoco import MjModel, MjData, mjx
    mj_model = MjModel.from_xml_path(MJCF_PATH)
    # mj_data = MjData(mj_model)
    # mjx_model = mjx.put_model(mj_model)
    # mjx_data = mjx.put_data(mj_model, mj_data)
    # print(mjx_model.actuator_acc0)

    # -- gym
    # gym_env = GymGo2Env(MJCF_PATH) # same problem as the MjModel.from_xml_path directly
    # print(type(gym_env))
    

    pass
from pathlib import Path
from brax.io import mjcf
from brax import envs
# from brax.envs.wrappers import torch as torch_wrapper

from unitree_robot.common.cdds import get_all_cdds_topics
from unitree_robot.common.datastructure import NETWORK_INTERFACE
from unitree_robot.train.test_pytorch_training import Trainer
from unitree_robot.train.brax_env import CustomEnv


if __name__ == "__main__":

    # env = envs.get_environment(env_name="humanoid", backend="mjx")

    env = CustomEnv(mjcf_path="./external/mjcf-robot-files/go2/scene.xml")

    print(type(env))

    # env = gym_wrapper.VectorGymWrapper(env)
    # automatically convert between jax ndarrays and T tensors:
    # env = torch_wrapper.TorchWrapper(
    #     env,
    #     device="cpu"
    # )
    
    Trainer(env, device="cpu", network_hidden_size=128)
    

    pass
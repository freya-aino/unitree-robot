import random
from copy import copy, deepcopy
from os import path
from types import NoneType
from typing import Any, Dict, List, Union
from torch2jax import t2j
# import gymnasium as gym
import jax
import mujoco
import numpy as np
import torch as T
import torch.utils.dlpack as torch_dlpack
import jax.dlpack as jax_dlpack
# from gymnasium import spaces
# from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from mujoco import MjData, MjModel, mjx

# TODO
# from gym.envs.registration import load_env_plugins
# from gym.envs.registration import make, register, registry, spec
# load_env_plugins()
# register(
#     "unitree-go2-standing",
#     entry_point="unitree_robot.training.environments:Go2Env"
# )


class MujocoMjxEnv:  # (gym.Env):
    """Superclass for all MuJoCo MJX environments."""

    def __init__(
        self,
        model_path: str,
        sim_frames_per_step: int,
        mujoco_timestep: float,
        initial_noise_scale: float,
        num_parallel_environments: int,
        # width: int = 1920,
        # height: int = 1080,
        # enable_renderer: bool = False,
        # render_fps: int = 60,
    ):
        # -- load mujoco model
        assert path.exists(model_path), f"File {model_path} does not exist"
        self.mj_model = MjModel.from_xml_path(model_path)
        self.mj_data = MjData(self.mj_model)

        self.mj_model.opt.timestep = mujoco_timestep

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data_initial = mjx.put_data(self.mj_model, self.mj_data)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        self.jit_step = jax.jit(mjx.step)

        # -- set variables
        self.sim_frames_per_step = sim_frames_per_step
        self.initial_noise_scale = initial_noise_scale
        self.num_parallel_environments = num_parallel_environments
        self.observation_space_size = self.mj_data.qpos.shape[0] + self.mj_data.qvel.shape[0]
        self.action_space_size = self.mj_data.actuator_length.shape[0] # TODO: thi is actuator_length to get action size but may be something more reliable

        # -- set relevant control spaces, observation spaces and mujoco data
        self.action_range = self.mj_model.actuator_ctrlrange.copy().astype(np.float32).T

        # self.observation_space_size = self.get_observation_size()
        # self.action_space_size = self.get_action_size()

        # self.init_qpos = self.mj_data.qpos.ravel().copy()
        # self.init_qvel = self.mj_data.qvel.ravel().copy()


    def _get_observations(self) -> T.Tensor:
        obs = jax.numpy.concatenate([self.mjx_data.qpos.copy(), self.mjx_data.qvel.copy()], axis=-1, dtype=jax.numpy.float32)
        return torch_dlpack.from_dlpack(obs).unsqueeze(1)

    # def get_observation_size(self) -> int:
    #     return self._get_observation().shape[0]
    #
    # def get_action_size(self) -> int:
    #     return T.Tensor(self.action_space.sample()).shape[0]
    #
    # def _get_observation(self) -> T.Tensor:
    #     raise NotImplementedError

    def step(self, action: T.Tensor) -> T.Tensor:

        action = jax_dlpack.from_dlpack(action.detach()).squeeze() / self.sim_frames_per_step

        # action_batch = t2j(action_batch)
        self.mjx_data = self.mjx_data.replace(ctrl=action)

        for i in range(self.sim_frames_per_step):
            self.mjx_data = jax.vmap(self.jit_step, in_axes=(None, 0))(self.mjx_model, self.mjx_data)

        return self._get_observations()

    def reset(self, seed: int):

        rng = jax.random.PRNGKey(seed)
        rng = jax.random.split(rng, self.num_parallel_environments)
        self.mjx_data = jax.vmap(lambda r: self.mjx_data.replace(qpos=jax.random.normal(key=r, shape=self.mjx_data.qpos.shape) * self.initial_noise_scale))(rng)

        return self._get_observations()

    # def render(self):
    #     assert self.renderer_enabled, (
    #         "trying to call render without renderer, pass 'enable_renderer=True' to the environment"
    #     )
    #     self.viewer.render("human")

    # def step(self, action: T.Tensor) -> T.Tensor:
    #     self._do_simulation(ctrl=action)
    #     obs = self._get_observation()
    #     return obs

    # def _do_simulation(self, ctrl: T.Tensor):
    #     """
    #     Step the simulation n number of frames and applying a control action.
    #     """
    #     # Check control input is contained in the action space
    #     assert ctrl.shape == self.action_space.shape, (
    #         f"Action dimension mismatch, expected {self.action_space_size}, got {ctrl.shape}"
    #     )

    #     # ctrl = ctrl.detach().cpu().numpy() / self.sim_frames_per_step

    #     # -- step mujoco simulation
    #     # self.data.ctrl[:] = ctrl
    #     # mujoco.mj_step(self.model, self.data, nstep=self.sim_frames_per_step)
    #     self.mjx_data = self.jit_step(self.mjx_model, self.mjx_data)
    #     # mujoco.mj_forward(self.model, self.data)

    #     # As of MuJoCo 2.0, force-related quantities like cacc are not computed
    #     # unless there's a force sensor in the model.
    #     # See https://github.com/openai/gym/issues/1541
    #     # mujoco.mj_rnePostConstraint(self.model, self.data)


class Go2EnvMJX(MujocoMjxEnv):
    def __init__(self, **kwargs):
        self.actuator_names = [
            "FL_calf", "FL_hip", "FL_thigh",
            "FR_calf", "FR_hip", "FR_thigh",
            "RL_calf", "RL_hip", "RL_thigh",
            "RR_calf", "RR_hip", "RR_thigh",
        ]

        # self.sensor_names = [
        #     "FL_calf_pos",
        #     "FL_calf_torque",
        #     "FL_calf_vel",
        #     "FL_hip_pos",
        #     "FL_hip_torque",
        #     "FL_hip_vel",
        #     "FL_thigh_pos",
        #     "FL_thigh_torque",
        #     "FL_thigh_vel",
        #     "FR_calf_pos",
        #     "FR_calf_torque",
        #     "FR_calf_vel",
        #     "FR_hip_pos",
        #     "FR_hip_torque",
        #     "FR_hip_vel",
        #     "FR_thigh_pos",
        #     "FR_thigh_torque",
        #     "FR_thigh_vel",
        #     "RL_calf_pos",
        #     "RL_calf_torque",
        #     "RL_calf_vel",
        #     "RL_hip_pos",
        #     "RL_hip_torque",
        #     "RL_hip_vel",
        #     "RL_thigh_pos",
        #     "RL_thigh_torque",
        #     "RL_thigh_vel",
        #     "RR_calf_pos",
        #     "RR_calf_torque",
        #     "RR_calf_vel",
        #     "RR_hip_pos",
        #     "RR_hip_torque",
        #     "RR_hip_vel",
        #     "RR_thigh_pos",
        #     "RR_thigh_torque",
        #     "RR_thigh_vel",
        #     "frame_pos",
        #     "frame_vel",
        #     # 'imu_acc',
        #     # 'imu_gyro',
        #     # 'imu_quat'
        # ]

        super().__init__(**kwargs)



# class MujocoEnv(gym.Env):
#     """Superclass for all MuJoCo environments."""
#
#     def __init__(
#         self,
#         model_path: str,
#         sim_frames_per_step: int,
#         camera_name: str,
#         mujoco_timestep: float,
#         initial_noise_scale: float,
#         width: int = 1920,
#         height: int = 1080,
#         enable_renderer: bool = False,
#         render_fps: int = 60,
#     ):
#         # -- metadata
#         # self.metadata["render_modes"] = ["human"]
#         # self.metadata["render_fps"] = render_fps
#         self.metadata["torch"] = True
#
#         # -- load mujoco model
#         assert path.exists(model_path), f"File {model_path} does not exist"
#         self.model = MjModel.from_xml_path(model_path)
#         self.data = MjData(self.model)
#
#         # -- set variables
#         self.model.opt.timestep = mujoco_timestep
#
#         # -- visualization
#         # "render_fps": int(np.round(1.0 / self.dt)), # TODO
#         self.renderer_enabled = enable_renderer
#         if self.renderer_enabled:
#             self.viewer = MujocoRenderer(
#                 model=self.model,
#                 data=self.data,
#                 # default_cam_config=DEFAULT_CAMERA_CONFIG,
#                 width=width,
#                 height=height,
#                 camera_name=camera_name,
#             )
#
#         # -- set relevant control spaces, observation spaces and mujoco data
#         ctrl_range = self.model.actuator_ctrlrange.copy().astype(np.float32)
#         low, high = ctrl_range.T
#         self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
#
#         self.observation_space_size = self.get_observation_size()
#         self.action_space_size = self.get_action_size()
#
#         self.init_qpos = self.data.qpos.ravel().copy()
#         self.init_qvel = self.data.qvel.ravel().copy()
#
#         # -- variables
#         self.width = width
#         self.height = height
#         self.sim_frames_per_step = sim_frames_per_step
#         self.camera_name = camera_name
#         self.initial_noise_scale = initial_noise_scale
#
#         super().__init__()
#
#     def get_observation_size(self) -> int:
#         return self._get_observation().shape[0]
#
#     def get_action_size(self) -> int:
#         return T.Tensor(self.action_space.sample()).shape[0]
#
#     def _get_observation(self) -> T.Tensor:
#         raise NotImplementedError
#
#     def reset(self, seed: int | NoneType):
#         # set seed
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
#             T.random.manual_seed(seed)
#
#         mujoco.mj_resetData(self.model, self.data)
#
#         # apply initial noise
#         self.data.qpos += np.random.normal(
#             scale=self.initial_noise_scale, size=self.init_qpos.shape
#         ).astype(np.float32)
#         self.data.qvel += np.random.normal(
#             scale=self.initial_noise_scale, size=self.init_qvel.shape
#         ).astype(np.float32)
#
#         # self.set_stat(self.init_qpos, self.init_qvel)
#         return self._get_observation()
#
#     def render(self):
#         assert self.renderer_enabled, (
#             "trying to call render without renderer, pass 'enable_renderer=True' to the environment"
#         )
#         self.viewer.render("human")
#
#     def step(self, action: T.Tensor) -> tuple[T.Tensor, MjData]:
#         self._do_simulation(ctrl=action)
#         obs = self._get_observation()
#         return obs, copy(self.data)
#
#     def _do_simulation(self, ctrl: T.Tensor):
#         """
#         Step the simulation n number of frames and applying a control action.
#         """
#         # Check control input is contained in the action space
#         assert ctrl.shape == self.action_space.shape, (
#             f"Action dimension mismatch, expected {self.action_space_size}, got {ctrl.shape}"
#         )
#
#         ctrl = ctrl.detach().cpu().numpy() / self.sim_frames_per_step
#
#         # -- step mujoco simulation
#         self.data.ctrl[:] = ctrl
#         mujoco.mj_step(self.model, self.data, nstep=self.sim_frames_per_step)
#         # mujoco.mj_forward(self.model, self.data)
#
#         # As of MuJoCo 2.0, force-related quantities like cacc are not computed
#         # unless there's a force sensor in the model.
#         # See https://github.com/openai/gym/issues/1541
#         mujoco.mj_rnePostConstraint(self.model, self.data)
#
#
# class Go2Env(MujocoEnv):
#     """Superclass for MuJoCo environments."""
#
#     def __init__(self, **kwargs):
#         self.actuator_names = [
#             "FL_calf",
#             "FL_hip",
#             "FL_thigh",
#             "FR_calf",
#             "FR_hip",
#             "FR_thigh",
#             "RL_calf",
#             "RL_hip",
#             "RL_thigh",
#             "RR_calf",
#             "RR_hip",
#             "RR_thigh",
#         ]
#
#         # self.sensor_names = [
#         #     "FL_calf_pos",
#         #     "FL_calf_torque",
#         #     "FL_calf_vel",
#         #     "FL_hip_pos",
#         #     "FL_hip_torque",
#         #     "FL_hip_vel",
#         #     "FL_thigh_pos",
#         #     "FL_thigh_torque",
#         #     "FL_thigh_vel",
#         #     "FR_calf_pos",
#         #     "FR_calf_torque",
#         #     "FR_calf_vel",
#         #     "FR_hip_pos",
#         #     "FR_hip_torque",
#         #     "FR_hip_vel",
#         #     "FR_thigh_pos",
#         #     "FR_thigh_torque",
#         #     "FR_thigh_vel",
#         #     "RL_calf_pos",
#         #     "RL_calf_torque",
#         #     "RL_calf_vel",
#         #     "RL_hip_pos",
#         #     "RL_hip_torque",
#         #     "RL_hip_vel",
#         #     "RL_thigh_pos",
#         #     "RL_thigh_torque",
#         #     "RL_thigh_vel",
#         #     "RR_calf_pos",
#         #     "RR_calf_torque",
#         #     "RR_calf_vel",
#         #     "RR_hip_pos",
#         #     "RR_hip_torque",
#         #     "RR_hip_vel",
#         #     "RR_thigh_pos",
#         #     "RR_thigh_torque",
#         #     "RR_thigh_vel",
#         #     "frame_pos",
#         #     "frame_vel",
#         #     # 'imu_acc',
#         #     # 'imu_gyro',
#         #     # 'imu_quat'
#         # ]
#
#         super().__init__(**kwargs)
#
#         # TODO:
#         # def _set_observation_space(self):
#         #     self.observation_space = spaces.Space(
#         #         shape=[len(self.sensor_names)],
#         #         dtype=np.float32,
#         #     )
#         # self.observation_space_size = self.get_observation_size()
#
#     def _get_observation(self) -> T.Tensor:
#         return T.from_numpy(
#             np.concatenate([self.data.qpos.ravel(), self.data.qvel.ravel()]).astype(
#                 np.float32
#             ),
#         )
#         # return self.get_sensor_state_array()
#         # return self.observation_space.sample() # TODO
#
#     # def get_sensor_state_dict(self) -> Dict[str, T.Tensor]:
#     #     return {
#     #         n: T.Tensor(self.data.sensor(n).data).to(dtype=T.float32)
#     #         for n in self.sensor_names
#     #     }
#
#     # def get_sensor_state_array(self) -> T.Tensor:
#     #     return T.concat([*self.get_sensor_state_dict().values()]).to(dtype=T.float32)

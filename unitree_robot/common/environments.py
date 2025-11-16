import threading
import time
from os import path
from typing import Tuple
from functools import partial
import jax
import mujoco.viewer as mjv
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.utils.dlpack as torch_dlpack
import jax.dlpack as jax_dlpack
import jax.numpy as npx
from mujoco import MjData, MjModel, mjx, Renderer
from mujoco.mjx import Data as MjxData
from mujoco.mjx import Model as MjxModel


class MujocoMjxEnv:
    """Superclass for all MuJoCo MJX environments."""

    def __init__(
        self,
        model_path: str,
        sim_frames_per_step: int,
        mujoco_timestep: float,
        initial_noise_scale: float,
        num_parallel_environments: int,
        observation_size: int,
        action_size: int,
    ):
        # -- set variables
        self.sim_frames_per_step = sim_frames_per_step
        self.initial_noise_scale = initial_noise_scale
        self.num_parallel_environments = num_parallel_environments
        self.observation_size = observation_size
        self.action_size = action_size

        # -- load mujoco model and create data
        assert path.exists(model_path), f"File {model_path} does not exist"
        self.mj_model = MjModel.from_xml_path(model_path)

        # -- set mujoco model parameter
        self.mj_model.opt.timestep = mujoco_timestep # INFO - currently there is no reason to switch this on, so default is just "safer"

        self.mj_data = MjData(self.mj_model)

        # --- instantiate mjx model and data
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data_initial = mjx.put_data(self.mj_model, self.mj_data)
        self.mjx_data_current = mjx.put_data(self.mj_model, self.mj_data)

        # -- jit the step function for distribution
        self.jit_step = jax.jit(mjx.step)
        # self.jit_forward = jax.jit(mjx.forward) # relevant for forward kinematics

        # -- set relevant control spaces, observation spaces and mujoco data
        self.action_ranges = self.mjx_model.actuator_ctrlrange.copy().astype(npx.float32)

    def _get_observations(self) -> T.Tensor:
        obs = jax.numpy.concatenate([self.mjx_data_current.qpos.copy(), self.mjx_data_current.qvel.copy()], axis=-1, dtype=npx.float32)
        return torch_dlpack.from_dlpack(obs)

    def step(self, action: T.Tensor) -> T.Tensor:

        assert (action.min() >= -1).all() and (action.max() <= 1).all(), "all action values must be between -1 and 1"

        self.set_ctrl_(action)

        for i in range(self.sim_frames_per_step):
            self.mjx_data_current = jax.vmap(self.jit_step, in_axes=(None, 0))(self.mjx_model, self.mjx_data_current)

        return self._get_observations()

    def reset(self, seed: int):

        rng = jax.random.PRNGKey(seed)
        rng = jax.random.split(rng, self.num_parallel_environments)
        self.mjx_data_current = jax.vmap(lambda r: self.mjx_data_initial.replace(qpos=jax.random.normal(key=r, shape=self.mjx_data_initial.qpos.shape, dtype=npx.float32) * self.initial_noise_scale))(rng)

        return self._get_observations()

    def set_ctrl_(self, action: T.Tensor):
        # scale the input_magnitude by the number of sim steps per action step
        action = jax_dlpack.from_dlpack(action.detach()) / self.sim_frames_per_step

        # scale the input by the available ctrl range
        mean = self.action_ranges.mean(axis=-1)
        scale = (self.action_ranges.max(axis=-1) - self.action_ranges.min(axis=-1)) / 2
        action = mean + action * scale

        # replace in mjx_data
        self.mjx_data_current = self.mjx_data_current.replace(ctrl=action)
    
    def unroll_step(
        self,
        agent: T.nn.Module,
        observation: T.Tensor,
        eval: bool = False
    ):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim == 2:
            observation = observation.unsqueeze(1)

        action, logits = agent.get_action_and_logits(observation, eval=eval)

        next_observation = self.step(action=F.tanh(action.squeeze()))

        return next_observation, action, logits

    
    # TODO: change to mjx_data OR make a seperate reset, step function for mj_data for testing purposes and create a seperate mjc_data version of this function
    # @staticmethod
    # def get_all_env_metrics(mj_data: MjData):
    #     return {
    #         "qvel_abs_sum": np.abs(mj_data.qvel).sum(),
    #         "qacc_abs_sum": np.abs(mj_data.qacc).sum(),
    #         "cvel_abs_sum": np.abs(mj_data.cvel).sum(),
    #         "cfrc_ext_abs_sum": np.abs(mj_data.cfrc_ext).sum(),
    #         "xfrc_applied_abs_sum": np.abs(mj_data.xfrc_applied).sum(),
    #         "actuator_force_abs_sum": np.abs(mj_data.actuator_force).sum(),
    #         "ctrl_abs_sum": np.abs(mj_data.ctrl).sum(),
    #         "qfrc_actuator_abs_sum": np.abs(mj_data.qfrc_actuator).sum(),
    #         "qfrc_passive_abs_sum": np.abs(mj_data.qfrc_passive).sum(),
    #         "qfrc_constraint_abs_sum": np.abs(mj_data.qfrc_constraint).sum(),
    #     }


class MjxRenderer:

    def __init__(
        self,
        mjx_env: MujocoMjxEnv,
        render_width: int,
        render_height: int,
        render_fps: int
    ):

        self.env = mjx_env

        self.render_width = render_width
        self.render_height = render_height
        self.render_fps = render_fps


        # "render_fps": int(np.round(1.0 / self.dt)), # TODO
        self.mj_data = mjx.get_data(self.env.mj_model, self.env.mjx_data_initial)
        self.viewer = mjv.launch_passive(
            model=self.env.mj_model,
            data=self.mj_data,
        )
        self.lock = threading.Lock()
        self.end_viewer_event = threading.Event()

    @staticmethod
    def render_thread_(viewer, lock: threading.Lock, render_fps: int, end_viewer_event: threading.Event):
        while viewer.is_running() and not end_viewer_event.is_set():
            start_time = time.perf_counter()
            lock.acquire()
            viewer.sync()
            lock.release()
            time.sleep((time.perf_counter() - start_time) / render_fps)

    def start_viewer(self):
        self.end_viewer_event.clear()
        viewer_thread = threading.Thread(
            target=self.render_thread_,
            args=(self.viewer, self.lock, self.render_fps, self.end_viewer_event)
        )
        viewer_thread.start()

    def stop_viewer(self):
        self.end_viewer_event.set()

    def update(self):
        self.lock.acquire()
        self.mj_data = mjx.get_data(self.env.mj_model, self.env.mjx_data_current)[0]
        self.lock.release()





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

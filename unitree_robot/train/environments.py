from copy import deepcopy, copy

import mujoco
import torch as T
import gymnasium as gym
from os import path
from numpy import float32 as np_float32
from typing import Any, Dict, Union, List
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from mujoco import MjData, MjModel


# TODO
# from gym.envs.registration import load_env_plugins
# from gym.envs.registration import make, register, registry, spec
# load_env_plugins()
# register(
#     "unitree-go2-standing",
#     entry_point="unitree_robot.training.environments:Go2Env"
# )


# DEFAULT_CAMERA_CONFIG = {
#     "trackbodyid": 1,
#     "distance": 4.0,
#     # "lookat": np.array((0.0, 0.0, 2.0)),
#     "elevation": -20.0,
# }




class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(
        self,
        model_path: str,
        sim_frames_per_step: int,
        camera_name: str,
        width: int = 1920,
        height: int = 1080,
        render_fps: int = 60,
    ):
        # -- metadata
        self.metadata["render_modes"] = ["human"]
        self.metadata["render_fps"] = render_fps
        self.metadata["torch"] = True

        # -- load mujoco model
        self.model_path = model_path
        assert path.exists(self.model_path), f"File {self.model_path} does not exist"

        self.model = MjModel.from_xml_path(self.model_path)
        self.data = MjData(self.model)

        # -- visualization
        # "render_fps": int(np.round(1.0 / self.dt)), # TODO
        self.viewer = MujocoRenderer(
            model=self.model,
            data=self.data,
            # default_cam_config=DEFAULT_CAMERA_CONFIG,
            width=width,
            height=height,
            camera_name=camera_name,
        )

        # -- set relevant control spaces, observation spaces and mujoco data
        # self.observation_space = observation_space
        ctrl_range = self.model.actuator_ctrlrange.copy().astype(np_float32)
        low, high = ctrl_range.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np_float32)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # -- variables
        self.width = width
        self.height = height
        self.sim_frames_per_step = sim_frames_per_step
        self.camera_name = camera_name

        super().__init__()

    def get_observation_size(self) -> int:
        return T.flatten(self._get_observation()).shape[0]

    def get_action_size(self) -> int:
        return T.flatten(T.Tensor(self.action_space.sample())).shape[0]

    def _get_observation(self):
        raise NotImplementedError

    def reset(self, seed: int):
        # assert self.observation_space, "observation space not set"
        mujoco.mj_resetData(self.model, self.data)
        # TODO: apply initial noise
        # self.set_stat(self.init_qpos, self.init_qvel)
        return super().reset(seed=seed)

    def render(self):
        self.viewer.render("human")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step(self, action: T.Tensor) -> tuple[T.Tensor, MjData]:
        # assert self.observation_space, "observation space not set"

        self._do_simulation(ctrl=action)
        # obs = self._get_observation()
        obs = self._get_observation()

        # super().step()
        return obs, copy(self.data)

    def _do_simulation(self, ctrl: T.Tensor):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        assert ctrl.shape == self.action_space.shape, "Action dimension mismatch"

        ctrl = ctrl.detach().cpu().numpy() / self.sim_frames_per_step

        # -- step mujoco simulation
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.sim_frames_per_step)
        # mujoco.mj_forward(self.model, self.data)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)


class Go2Env(MujocoEnv):
    """Superclass for MuJoCo environments."""

    def __init__(
        self, model_path: str, sim_frames_per_step: int, camera_name: str = "main"
    ):
        self.actuator_names = [
            "FL_calf",
            "FL_hip",
            "FL_thigh",
            "FR_calf",
            "FR_hip",
            "FR_thigh",
            "RL_calf",
            "RL_hip",
            "RL_thigh",
            "RR_calf",
            "RR_hip",
            "RR_thigh",
        ]

        self.sensor_names = [
            "FL_calf_pos",
            "FL_calf_torque",
            "FL_calf_vel",
            "FL_hip_pos",
            "FL_hip_torque",
            "FL_hip_vel",
            "FL_thigh_pos",
            "FL_thigh_torque",
            "FL_thigh_vel",
            "FR_calf_pos",
            "FR_calf_torque",
            "FR_calf_vel",
            "FR_hip_pos",
            "FR_hip_torque",
            "FR_hip_vel",
            "FR_thigh_pos",
            "FR_thigh_torque",
            "FR_thigh_vel",
            "RL_calf_pos",
            "RL_calf_torque",
            "RL_calf_vel",
            "RL_hip_pos",
            "RL_hip_torque",
            "RL_hip_vel",
            "RL_thigh_pos",
            "RL_thigh_torque",
            "RL_thigh_vel",
            "RR_calf_pos",
            "RR_calf_torque",
            "RR_calf_vel",
            "RR_hip_pos",
            "RR_hip_torque",
            "RR_hip_vel",
            "RR_thigh_pos",
            "RR_thigh_torque",
            "RR_thigh_vel",
            "frame_pos",
            "frame_vel",
            # 'imu_acc',
            # 'imu_gyro',
            # 'imu_quat'
        ]

        # self._set_observation_space()

        super().__init__(
            model_path=model_path,
            sim_frames_per_step=sim_frames_per_step,
            camera_name=camera_name,
        )

        # TODO:
        # def _set_observation_space(self):
        #     self.observation_space = spaces.Space(
        #         shape=[len(self.sensor_names)],
        #         dtype=np.float32,
        #     )
        self.observation_space_size = self.get_observation_size()

    def _get_observation(self) -> T.Tensor:
        return self.get_sensor_state_array()
        # return self.observation_space.sample() # TODO

    def get_sensor_state_dict(self) -> Dict[str, T.Tensor]:
        return {
            n: T.Tensor(self.data.sensor(n).data).to(dtype=T.float32)
            for n in self.sensor_names
        }

    def get_sensor_state_array(self) -> T.Tensor:
        return T.concat([*self.get_sensor_state_dict().values()]).to(dtype=T.float32)



class MultiMujocoEnv:

    def __init__(
        self, 
        env: MujocoEnv,
        copies: int
    ):
        self.envs = [deepcopy(env) for _ in range(copies)]

    def step(self, actions):
        for e in self.envs:
            e.step(action=action)
        
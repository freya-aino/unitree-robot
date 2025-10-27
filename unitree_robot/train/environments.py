import mujoco
from numpy.typing import NDArray
import torch as T
import gymnasium as gym

from os import path
from typing import Union, Optional, Any
# from brax import base
# from brax import math
# from brax.envs.base import PipelineEnv, State
# from brax.io import mjcf
# from brax.envs.wrappers.torch import TorchWrapper
# from jax import numpy as jnp

# from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Space, Box
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, WindowViewer

import numpy as np

# from gym.envs.registration import load_env_plugins
# from gym.envs.registration import make, register, registry, spec

# load_env_plugins()

# register(
#     "unitree-go2-standing",
#     entry_point="unitree_robot.training.environments:Go2Env"
# )

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

DEFAULT_SIZE = 480


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(
        self,
        model_path,
        sim_frames_per_step: int,
        observation_space: Space,
        # render_mode: Optional[str] = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        
        # -- variables
        self.width = width
        self.height = height
        self.sim_frames_per_step = sim_frames_per_step


        # -- load mujoco model
        self.model_path = model_path
        assert path.exists(self.model_path), f"File {self.model_path} does not exist"
        
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        # MjrContext will copy model.vis.global_.off* to con.off*
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)


        # -- visualization
        # self.viewer = WindowViewer(model=self.model, data=self.data, width=self.width, height=self.height)

        # from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
        # self.viewer = WindowsError(
        #     self.model,
        #     self.data
        # )
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        self.viewer = MujocoRenderer(
            self.model, 
            self.data, 
            DEFAULT_CAMERA_CONFIG,
            width,
            height,
            # camera_id=camera_id,
            # camera_name=camera_name
        )

        # -- set relevant control spaces, observation spaces and mujoco data
        self.observation_space = observation_space
        self._set_action_space()
        
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()



        # -- TODO: there is probably a way to simplify this
        # assert self.metadata["render_modes"] == [
        #     "human",
        #     "rgb_array",
        #     "depth_array",
        # ], self.metadata["render_modes"]
        # assert (
        #     int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        # ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.camera_name = camera_name
        self.camera_id = camera_id

        super().__init__()

    @property
    def dt(self):
        return self.model.opt.timestep * self.sim_frames_per_step

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    # methods to override:
    # ----------------------------

    # def reset_model(self):
    #     """
    #     Reset the robot degrees of freedom (qpos and qvel).
    #     Implement this in each subclass.
    #     """
    #     raise NotImplementedError

    # def _reset_simulation(
    #     self,
    #     seed: Optional[int] = None,
    #     options: Optional[dict] = None,
    # ):
    #     """
    #     Reset MuJoCo simulation data structures, mjModel and mjData.
    #     """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        mujoco.mj_resetData(self.model, self.data)
        return super().reset(seed=seed, options=options)

    def render(self):
        self.viewer.render("human")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def set_state(self, qpos: NDArray[np.float32], qvel: NDArray[np.float32]):
        """
        Set the joints position qpos and velocity qvel of the model. Override this method depending on the MuJoCo bindings used.
        """
        
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)

        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)
        if self.model.na == 0:
            self.data.act[:] = None
        mujoco.mj_forward(self.model, self.data)

    def do_simulation(self, ctrl: np.ndarray, n_frames: int):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        assert ctrl.shape == self.action_space.shape, "Action dimension mismatch"
        
        # -- step mujoco simulation
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.sim_frames_per_step)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)


        # # TODO: I think for renderer, not sure
        # mujoco.mjv_updateScene(self.model, self.data)


    # def get_body_com(self, body_name):
    #     """Return the cartesian position of a body frame"""
    #     return self.data.body(body_name).xpos

    # def get_state_vector(self):
    #     """Return the position and velocity joint states of the model"""
    #     return self.data.qpos, self.data.qvel


class Go2Env(MujocoEnv):
    """Superclass for MuJoCo environments."""

    def __init__(
        self,
        model_path: str,
        sim_frames_per_step: int,
        seed: int,
    ):
        
        self.actuator_names = [
            'FL_calf', 'FL_hip', 'FL_thigh', 
            'FR_calf', 'FR_hip', 'FR_thigh', 
            'RL_calf', 'RL_hip', 'RL_thigh', 
            'RR_calf', 'RR_hip', 'RR_thigh'
        ]

        self.sensor_names = [
            'FL_calf_pos', 'FL_calf_torque', 'FL_calf_vel', 
            'FL_hip_pos', 'FL_hip_torque', 'FL_hip_vel', 
            'FL_thigh_pos', 'FL_thigh_torque', 'FL_thigh_vel', 
            'FR_calf_pos', 'FR_calf_torque', 'FR_calf_vel', 
            'FR_hip_pos', 'FR_hip_torque', 'FR_hip_vel', 
            'FR_thigh_pos', 'FR_thigh_torque', 'FR_thigh_vel', 
            'RL_calf_pos', 'RL_calf_torque', 'RL_calf_vel', 
            'RL_hip_pos', 'RL_hip_torque', 'RL_hip_vel', 
            'RL_thigh_pos', 'RL_thigh_torque', 'RL_thigh_vel', 
            'RR_calf_pos', 'RR_calf_torque', 'RR_calf_vel', 
            'RR_hip_pos', 'RR_hip_torque', 'RR_hip_vel', 
            'RR_thigh_pos', 'RR_thigh_torque', 'RR_thigh_vel', 
            'frame_pos', 'frame_vel', 
            'imu_acc', 'imu_gyro', 'imu_quat'
        ]


        # self.action_space_size = len(self.actuator_names)
        # self.observation_space_size = len(self.sensor_names)
        # TODO
        observation_space = Space(
            shape = [len(self.sensor_names)],
            dtype=np.float32,
            seed=seed
        )

        super().__init__(
            model_path=model_path,
            sim_frames_per_step=sim_frames_per_step,
            observation_space=observation_space,
            # render_mode="human" # TODO: figure out what human means
        )

    def get_sensor_state(self):
        return {n: self.data.sensor(n).data for n in self.sensor_names}
        


# class GymGo2Env(MujocoEnv, EzPickle):
#     def __init__(
#         self,
#         xml_file: str,
#         sim_frames_per_step=5, # number of mujoco simulation steps for each gym step
#         default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
#         # forward_reward_weight: float = 1.25,
#         # ctrl_cost_weight: float = 0.1,
#         # contact_cost_weight: float = 5e-7,
#         # contact_cost_range: tuple[float, float] = (-np.inf, 10.0),
#         # healthy_reward: float = 5.0,
#         # terminate_when_unhealthy: bool = True,
#         # healthy_z_range: tuple[float, float] = (1.0, 2.0),
#         # reset_noise_scale: float = 1e-2,
#         # exclude_current_positions_from_observation: bool = True,
#         # include_cinert_in_observation: bool = True,
#         # include_cvel_in_observation: bool = True,
#         # include_qfrc_actuator_in_observation: bool = True,
#         # include_cfrc_ext_in_observation: bool = True,
#         **kwargs,
#     ):
#         EzPickle.__init__(
#             self,
#             xml_file,
#             sim_frames_per_step,
#             default_camera_config,
#             # forward_reward_weight,
#             # ctrl_cost_weight,
#             # contact_cost_weight,
#             # contact_cost_range,
#             # healthy_reward,
#             # terminate_when_unhealthy,
#             # healthy_z_range,
#             # reset_noise_scale,
#             # exclude_current_positions_from_observation,
#             # include_cinert_in_observation,
#             # include_cvel_in_observation,
#             # include_qfrc_actuator_in_observation,
#             # include_cfrc_ext_in_observation,
#             **kwargs,
#         )

#         MujocoEnv.__init__(
#             self,
#             xml_file,
#             sim_frames_per_step,
#             observation_space=Space,
#             render_mode="human",
#             **kwargs,
#         )
        
#         self.metadata = {
#             "render_modes": [
#                 "human",
#                 "rgb_array",
#                 "depth_array",
#                 "rgbd_tuple",
#             ],
#             "render_fps": int(np.round(1.0 / self.dt)),
#         }
    
#     def step(self, action: np.ndarray[base.Any, np.dtype[np.floating[np._32Bit]]]) -> tuple[ndarray[Any, dtype[floating[_64Bit]]], floating[_64Bit], bool, bool, dict[str, floating[_64Bit]]]:
#         return super().step(action)
    
#     def reset(self, *, seed: int | None = None, options: dict | None = None):
#         return super().reset(seed=seed, options=options)
    
#     def render(self):
#         return super().render()
    

# class Go2BraxEnv(PipelineEnv):

#     def __init__(self, sys: base.System, backend: str = 'mjx', n_frames: int = 1, debug: bool = False):
#         super().__init__(sys, backend, n_frames, debug)
    
#     def step(self, state: State, action: jax.Array) -> State:
#         return super().step(state, action)
    
#     def reset(self, rng: jax.Array) -> State:
#         return super().reset(rng)
    


# class Go2Environment(gym.Env):

#     def __init__(self, mjcf_path: str):


#         model = mjcf.load_mjmodel(mjcf_path)
#         sys = mjx.put_model(model)
#         sys = TorchWrapper(model, device=device)

#         # # override menagerie params for smoother policy
#         # sys = sys.replace(
#         #     dof_damping=sys.dof_damping.at[6:].set(0.5239),
#         #     actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
#         #     actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
#         # )

#         sys = sys.tree_replace({
#             'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
#             'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
#             "opt.timestep": 1/250,
#             'opt.iterations': 1,
#             'opt.ls_iterations': 4,
#         })

#         self._dt = 1/50
#         n_frames = self._dt // sys.opt.timestep

#         super().__init__(sys=sys, backend="mjx", n_frames=n_frames)


#     def reset(self, rng: jax.Array) -> State:
#         key, theta_key, qd_key = jax.random.split(rng, 3)

#         theta_init = jax.random.uniform(theta_key, (1,), minval=-0.1, maxval=0.1)[0]
        
#         # q structure: [x th]
#         q_init = jnp.array([0.0, theta_init])
        
#         # qd structure: [dx dth]
#         qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)        
        
#         # Initialize State:
#         pipeline_state = self.pipeline_init(q_init, qd_init)

#         # Initialize Rewards:
#         reward, done = jnp.zeros(2)

#         # Get observation for RL Algorithm (Input to our neural net):
#         observation = self.get_observation(pipeline_state)

#         # Metrics:
#         metrics = {
#             'rewards': reward,
#             'observation': observation,
#         }

#         state = State(
#             pipeline_state=pipeline_state,
#             obs=observation,
#             reward=reward,
#             done=done,
#             metrics=metrics,
#         )

#         return state
    

#     def step(self, state: State, action: jax.Array) -> State:
#         """Run one timestep of the environment's dynamics."""


#         # -- Get the current pipeline state and do the action to get the next step ss well
#         pipeline_state_t_0 = state.pipeline_state
#         assert pipeline_state_t_0 is not None
#         pipeline_state_t_1 = self.pipeline_step(pipeline_state_t_0, action)


#         # velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
#         # forward_reward = velocity[0]

#         # min_z, max_z = self._healthy_z_range
#         # is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
#         # is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
#         # if self._terminate_when_unhealthy:
#         #   healthy_reward = self._healthy_reward
#         # else:
#         #   healthy_reward = self._healthy_reward * is_healthy
#         # ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
#         # contact_cost = 0.0

#         obs = self._get_obs(pipeline_state_t_1)

#         # -- sum up reward
#         reward = jp.zeros(1) # TODO

#         # -- check if "done"
#         done = False

#         # -- update state metrics
#         state.metrics.update(reward=reward)
#         # state.metrics.update(
#         #     reward_forward=forward_reward,
#         #     reward_survive=healthy_reward,
#         #     reward_ctrl=-ctrl_cost,
#         #     reward_contact=-contact_cost,
#         #     x_position=pipeline_state.x.pos[0, 0],
#         #     y_position=pipeline_state.x.pos[0, 1],
#         #     distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
#         #     x_velocity=velocity[0],
#         #     y_velocity=velocity[1],
#         #     forward_reward=forward_reward,
#         # )

#         return state.replace(
#             pipeline_state=pipeline_state_t_1, 
#             obs=obs, 
#             reward=reward, 
#             # done=done,
#         )


#     def _get_obs(self, pipeline_state: base.State) -> jax.Array:
#         """Observe body position and velocities."""
#         qpos = pipeline_state.q
#         qvel = pipeline_state.qd

#         # exclude current observaion from state
#         qpos = pipeline_state.q[2:]

#         return jp.concatenate([qpos] + [qvel])
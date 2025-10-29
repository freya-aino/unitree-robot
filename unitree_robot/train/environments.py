import mujoco
import torch as T
import gymnasium as gym
import quaternion

from os import path
from typing import Union, Optional, Any, Dict
from numpy.typing import NDArray
# from jax import numpy as jnp

from gymnasium import spaces
# from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from mujoco import MjData, MjModel

import numpy as np
import scipy

from scipy.spatial.transform import Rotation as R

from unitree_robot.train.experiments import Experiment

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
        experiment: Experiment,
        sim_frames_per_step: int,
        camera_name: str,
        width: int = 1920,
        height: int = 1080,
        render_fps: int = 60
    ):

        # -- metadata
        self.metadata["render_modes"] = ["human"]
        self.metadata["render_fps"] = render_fps
        self.metadata["torch"] = True

        # -- load mujoco model
        self.model_path = model_path
        assert path.exists(self.model_path), f"File {self.model_path} does not exist"
        
        self.model = MjModel.from_xml_path(self.model_path)
        # MjrContext will copy model.vis.global_.off* to con.off*
        # self.model.vis.global_.offwidth = self.width
        # self.model.vis.global_.offheight = self.height
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
        ctrl_range = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = ctrl_range.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # -- variables
        self.width = width
        self.height = height
        self.sim_frames_per_step = sim_frames_per_step
        self.camera_name = camera_name
        self.experiment = experiment

        super().__init__()

    def _set_observation_space(self):
        raise NotImplementedError

    def _get_observation(self):
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        assert self.observation_space, "observation space not set"

        # mujoco.mj_resetData(self.model, self.data)
        # TODO: apply initial noise
        # self.set_stat(self.init_qpos, self.init_qvel)
        return super().reset(seed=seed, options=options)

    def render(self):
        self.viewer.render("human")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step(self, action: Any) -> tuple[NDArray[np.float32], Dict[str, NDArray]]:
        assert self.observation_space, "observation space not set"

        self._do_simulation(ctrl=action)
        obs = self._get_observation()

        return obs, self.experiment.calculate_reward(mj_data=self.data)
        # return  super().step() # TODO: dont know how important the step() function is

    def _do_simulation(self, ctrl: np.ndarray):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        assert ctrl.shape == self.action_space.shape, "Action dimension mismatch"

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
        self,
        model_path: str,
        sim_frames_per_step: int,
        experiment: Experiment,
        camera_name: str = "main"
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
            'imu_acc', 
            'imu_gyro', 
            # 'imu_quat'
        ]

        self._set_observation_space()

        super().__init__(
            model_path=model_path,
            sim_frames_per_step=sim_frames_per_step,
            camera_name=camera_name,
            experiment=experiment,
        )

    def _set_observation_space(self):
        self.observation_space = spaces.Space(
            shape=[len(self.sensor_names)],
            dtype=np.float32,
        )

    def _get_observation(self) -> NDArray[np.float32]:
        # return self.get_sensor_state_array()
        return np.random.random(size=[len(self.sensor_names)]).astype(np.float32)
        # return self.observation_space.sample() # TODO

    def get_sensor_state_dict(self) -> Dict[str, NDArray[np.float32]]:
        return {n: self.data.sensor(n).data for n in self.sensor_names}
        
    def get_sensor_state_array(self) -> NDArray[np.float32]:
        return np.concatenate([self.data.sensor(n).data for n in self.sensor_names], dtype=np.float32)

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
import torch as T
import gymnasium as gym

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jnp
from mujoco import MjModel, MjData



class Go2Environment(gym.Env):

    def __init__(self, mjcf_path: str):

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

        self.action_space_size = len(self.actuator_names)
        self.observation_space_size = len(self.sensor_names)


        # model = mjcf.load_mjmodel(mjcf_path)
        # sys = mjx.put_model(model)

        # # override menagerie params for smoother policy
        # sys = sys.replace(
        #     dof_damping=sys.dof_damping.at[6:].set(0.5239),
        #     actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        #     actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        # )

        sys = sys.tree_replace({
            'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
            'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            "opt.timestep": 1/250,
            'opt.iterations': 1,
            'opt.ls_iterations': 4,
        })

        self._dt = 1/50
        n_frames = self._dt // sys.opt.timestep

        super().__init__(sys=sys, backend="mjx", n_frames=n_frames)


    def reset(self, rng: jax.Array) -> State:
        key, theta_key, qd_key = jax.random.split(rng, 3)

        theta_init = jax.random.uniform(theta_key, (1,), minval=-0.1, maxval=0.1)[0]
        
        # q structure: [x th]
        q_init = jnp.array([0.0, theta_init])
        
        # qd structure: [dx dth]
        qd_init = jax.random.uniform(qd_key, (2,), minval=-0.1, maxval=0.1)        
        
        # Initialize State:
        pipeline_state = self.pipeline_init(q_init, qd_init)

        # Initialize Rewards:
        reward, done = jnp.zeros(2)

        # Get observation for RL Algorithm (Input to our neural net):
        observation = self.get_observation(pipeline_state)

        # Metrics:
        metrics = {
            'rewards': reward,
            'observation': observation,
        }

        state = State(
            pipeline_state=pipeline_state,
            obs=observation,
            reward=reward,
            done=done,
            metrics=metrics,
        )

        return state
    

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""


        # -- Get the current pipeline state and do the action to get the next step ss well
        pipeline_state_t_0 = state.pipeline_state
        assert pipeline_state_t_0 is not None
        pipeline_state_t_1 = self.pipeline_step(pipeline_state_t_0, action)


        # velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        # forward_reward = velocity[0]

        # min_z, max_z = self._healthy_z_range
        # is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        # is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        # if self._terminate_when_unhealthy:
        #   healthy_reward = self._healthy_reward
        # else:
        #   healthy_reward = self._healthy_reward * is_healthy
        # ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        # contact_cost = 0.0

        obs = self._get_obs(pipeline_state_t_1)

        # -- sum up reward
        reward = jp.zeros(1) # TODO

        # -- check if "done"
        done = False

        # -- update state metrics
        state.metrics.update(reward=reward)
        # state.metrics.update(
        #     reward_forward=forward_reward,
        #     reward_survive=healthy_reward,
        #     reward_ctrl=-ctrl_cost,
        #     reward_contact=-contact_cost,
        #     x_position=pipeline_state.x.pos[0, 0],
        #     y_position=pipeline_state.x.pos[0, 1],
        #     distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
        #     x_velocity=velocity[0],
        #     y_velocity=velocity[1],
        #     forward_reward=forward_reward,
        # )

        return state.replace(
            pipeline_state=pipeline_state_t_1, 
            obs=obs, 
            reward=reward, 
            # done=done,
        )


    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        # exclude current observaion from state
        qpos = pipeline_state.q[2:]

        return jp.concatenate([qpos] + [qvel])
import jax
import mujoco

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp
from mujoco import mjx


class CustomEnv(PipelineEnv):

  def __init__(self, mjcf_path: str):

    model = mjcf.load_mjmodel(mjcf_path)
    sys = mjx.put_model(model)
    # model = mjx.put_model(sys)
    # brax_system.actuator_acc0
    # brax_system.sensor()


    # set the action and observation shapes
    # self.action_size = [12]
    # self.observation_size = [12, 3]

    # from BarkourEnv from brax mujoco tutorial notebook
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

    # if backend == 'positional':
    #   sys = sys.replace(
    #       actuator=sys.actuator.replace(
    #           gear=200 * jp.ones_like(sys.actuator.gear)
    #       )
    #   )

    self._dt = 1/50

    n_frames = self._dt // sys.opt.timestep

    super().__init__(sys=sys, backend="mjx", n_frames=n_frames)

    # self._ctrl_cost_weight = ctrl_cost_weight
    # self._contact_cost_weight = contact_cost_weight
    # self._healthy_reward = healthy_reward
    # self._terminate_when_unhealthy = terminate_when_unhealthy
    # self._healthy_z_range = healthy_z_range
    # self._contact_force_range = contact_force_range
    # self._exclude_current_positions_from_observation = (
    #     exclude_current_positions_from_observation
    # )

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    q = self.sys.init_q + jax.random.uniform(
        rng1, (self.sys.q_size(),), minval=low, maxval=hi
    )
    qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

    pipeline_state = self.pipeline_init(q, qd)
    obs = self._get_obs(pipeline_state)

    reward, done, zero = jp.zeros(3)
    metrics = {
        "reward": zero
        # 'reward_forward': zero,
        # 'reward_survive': zero,
        # 'reward_ctrl': zero,
        # 'reward_contact': zero,
        # 'x_position': zero,
        # 'y_position': zero,
        # 'distance_from_origin': zero,
        # 'x_velocity': zero,
        # 'y_velocity': zero,
        # 'forward_reward': zero,
    }
    return State(pipeline_state, obs, reward, done, metrics)


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
    state.metrics.update(
        reward=reward
    )
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
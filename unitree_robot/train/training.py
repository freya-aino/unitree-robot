import collections
import functools
import math
import os
import time
import jax
# import brax
import numpy as np
import torch as T
import torch.nn.functional as F

from torch import nn
from torch import optim
from mujoco import mjx
from datetime import datetime
from typing import Any, Callable, Dict, Optional
# from brax.envs.wrappers import gym as gym_wrapper
# from brax.envs.wrappers.torch import TorchWrapper
# from brax.io import metrics, mjcf
# from brax.envs.base import PipelineEnv
# from brax.training.agents.ppo import train as ppo

from unitree_robot.common.networks import BasicPolicyValueNetwork
from unitree_robot.common.datastructure import UnrollData
from unitree_robot.train.environments import MujocoEnv

class PPOAgent(nn.Module):
  """Standard PPO Agent with GAE and observation normalization."""

  def __init__(
      self,
      input_size: int, 
      output_size: int, 
      network_hidden_size: int, 
      entropy_cost: float, 
      discounting: float, 
      reward_scaling: float, 
      device: str
    ):

    super(PPOAgent, self).__init__()

    self.network = BasicPolicyValueNetwork(input_size, output_size, network_hidden_size)

    self.num_steps = T.zeros((), device=device)
    self.entropy_cost = entropy_cost
    self.discounting = discounting
    self.reward_scaling = reward_scaling
    self.lambda_ = 0.95
    self.epsilon = 0.3
    self.device = device

  @T.jit.export
  def dist_create(self, logits):
    """Normal followed by tanh.

    T.distribution doesn't work with T.jit, so we roll our own."""
    loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale) + .001
    return loc, scale

  @T.jit.export
  def dist_sample_no_postprocess(self, loc, scale):
    return T.normal(loc, scale)

  @T.jit.export
  def dist_entropy(self, loc, scale):
    log_normalized = 0.5 * math.log(2 * math.pi) + T.log(scale)
    entropy = 0.5 + log_normalized
    entropy = entropy * T.ones_like(loc)
    dist = T.normal(loc, scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    entropy = entropy + log_det_jacobian
    return entropy.sum(dim=-1)

  @T.jit.export
  def dist_log_prob(self, loc, scale, dist):
    log_unnormalized = -0.5 * ((dist - loc) / scale).square()
    log_normalized = 0.5 * math.log(2 * math.pi) + T.log(scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    log_prob = log_unnormalized - log_normalized - log_det_jacobian
    return log_prob.sum(dim=-1)

  @T.jit.export
  def update_normalization(self, observation):
    self.num_steps += observation.shape[0] * observation.shape[1]
    input_to_old_mean = observation - self.running_mean
    mean_diff = T.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
    self.running_mean = self.running_mean + mean_diff
    input_to_new_mean = observation - self.running_mean
    var_diff = T.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
    self.running_variance = self.running_variance + var_diff

  @T.jit.export
  def normalize(self, observation):
    variance = self.running_variance / (self.num_steps + 1.0)
    variance = T.clip(variance, 1e-6, 1e6)
    return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

  @T.jit.export
  def get_logits_action(self, observation):
    observation = self.normalize(observation)
    logits = self.network.policy_forward(observation)
    loc, scale = self.dist_create(logits)
    action = self.dist_sample_no_postprocess(loc, scale)
    return logits, action

  @T.jit.export
  def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = T.cat([values[1:], T.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = reward + self.discounting * (
        1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = T.zeros_like(bootstrap_value)
    vs_minus_v_xs = T.zeros_like(truncation_mask)

    for ti in range(truncation_mask.shape[0]):
      ti = truncation_mask.shape[0] - ti - 1
      acc = deltas[ti] + self.discounting * (
          1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
      vs_minus_v_xs[ti] = acc

    # Add V(x_s) to get v_s.
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = T.cat([vs[1:], T.unsqueeze(bootstrap_value, 0)], 0)
    advantages = (reward + self.discounting *
                  (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return vs, advantages

  @T.jit.export
  def loss(self, unroll_data: UnrollData):
    
    observation = self.normalize(unroll_data.observation)
    policy_logits = self.network.policy_forward(observation[:-1])
    baseline = self.network.value_forward(observation)
    baseline = T.squeeze(baseline, dim=-1)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = baseline[-1]
    baseline = baseline[:-1]
    reward = unroll_data.reward * self.reward_scaling
    termination = unroll_data.done * (1 - unroll_data.truncation)

    loc, scale = self.dist_create(unroll_data.logits)
    behaviour_action_log_probs = self.dist_log_prob(loc, scale, unroll_data.action)
    loc, scale = self.dist_create(policy_logits)
    target_action_log_probs = self.dist_log_prob(loc, scale, unroll_data.action)

    with T.no_grad():
      vs, advantages = self.compute_gae(
        truncation=unroll_data.truncation,
        termination=termination,
        reward=reward,
        values=baseline,
        bootstrap_value=bootstrap_value
      )

    rho_s = T.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
    policy_loss = -T.mean(T.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = T.mean(v_error * v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = T.mean(self.dist_entropy(loc, scale))
    entropy_loss = self.entropy_cost * -entropy

    return policy_loss + v_loss + entropy_loss
  

class Trainer:

  def __init__(
    self, 
    env: MujocoEnv,
    device: str,
    network_hidden_size: int,
    reward_scaling: float = .1,
    entropy_cost: float = 1e-2,
    discounting: float = .97,
    learning_rate: float = 3e-4,
    optimizer_fn=T.optim.AdamW,
    rng_seed: int = 0
  ) -> None:
    
    # -- set device
    if T.cuda.is_available() and "cuda" in device:
      print("Trainer: device set to gpu (cuda) !")
      self.device = device
    else:
      print("Trainer: device set to cpu !")
      self.device = "cpu"

    # -- variables
    # self.reward_shape = [1] # TODO: this is just for a simple summed reward, this could be a vector, calculate the shape here in the __init__
    # self.env = TorchWrapper(env, device=device)
    # self.env.reset(jax.random.key(rng_seed))
    
    
    # -- create agent
    self.agent = T.jit.script(
      PPOAgent(
        input_size=self.env.observation_space_size,
        output_size=self.env.action_space_size,
        network_hidden_size=network_hidden_size,
        entropy_cost=entropy_cost, 
        discounting=discounting, 
        reward_scaling=reward_scaling, 
        device=device
      ).to(device=device)
    )

    # -- set up optimizer
    self.optim = optimizer_fn(self.agent.parameters(), lr=learning_rate)


  # def eval_unroll(self, length):
  #   """Return number of episodes and average reward for a single unroll."""
  #   observation = self.env.reset()
  #   episodes = T.zeros((), device=self.device)
  #   episode_reward = T.zeros((), device=self.device)
  #   for _ in range(length):
  #     _, action = self.agent.get_logits_action(observation)
  #     observation, reward, done, _ = self.env.step(action)
  #     episodes += T.sum(done)
  #     episode_reward += T.sum(reward)
  #   return episodes, episode_reward / episodes

  def train_unroll(self, observation, num_unrolls, unroll_length):
    """Return step data over multple unrolls."""
    
    unrolls = UnrollData.initialize_empty(
      num_unrolls=num_unrolls, 
      unroll_length=unroll_length,
      observation_shape=self.observation_space_shape,
      action_shape=self.action_space_shape,
      reward_shape=self.reward_shape,
      device=self.device,
    )

    for i in range(num_unrolls):
      for j in range(unroll_length):
        
        # decide which action to take depending on the environment observation (robot state)
        logits, action = self.agent.get_logits_action(observation)

        # take a step in the environment and get its return values like the local reward for taking that action.
        observation, reward, done, info = self.env.step(action)
        
        unrolls.observation[i, j, :] = observation[:]
        unrolls.logits[i, j, :] = logits[:]
        unrolls.action[i,j, :] = action[:]
        unrolls.reward[i, j, :] = reward[:]
        unrolls.done[i,j] = done
        unrolls.truncation[i,j] = info['truncation']

    return observation, unrolls


  def train(
      self,
      device: str,
      epochs: int = 4,
      batch_size: int = 2048,
      unroll_length: int = 5,
      progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
  ):
    
    # env warmup
    self.env.reset()
    action = T.zeros(self.env.action_space_size).to(device)
    self.env.step(action)


    self.unroll_length = unroll_length

    sps = 0
    total_steps = 0
    total_loss = 0
    # for eval_i in range(eval_frequency + 1):
    # if progress_fn:
    #   t = time.time()
    #   with T.no_grad():
    #     episode_count, episode_reward = self.eval_unroll(episode_length)
    #   duration = time.time() - t
    #   # TODO(brax-team): only count stats from completed episodes
    #   episode_avg_length = self.env.num_envs * episode_length / episode_count
    #   eval_sps = self.env.num_envs * episode_length / duration
    #   progress = {
    #       'eval/episode_reward': episode_reward,
    #       'eval/completed_episodes': episode_count,
    #       'eval/avg_episode_length': episode_avg_length,
    #       'speed/sps': sps,
    #       'speed/eval_sps': eval_sps,
    #       'losses/total_loss': total_loss,
    #   }
    #   progress_fn(total_steps, progress)
    # if eval_i == eval_frequency:
    #   break

    observation = self.env.reset()
    # num_steps = batch_size * num_minibatches * unroll_length
    # num_epochs = num_timesteps // (num_steps * eval_frequency)
    # num_unrolls = batch_size * num_minibatches // self.env.num_envs

    t = time.time()
    for _ in range(epochs):

      observation, unroll_data = self.train_unroll(observation, self.unroll_length, unroll_length)

      unroll_data.observation = unroll_data.observation.reshape([
        unroll_data.observation.shape[0] + unroll_data.observation.shape[1],
        -1
      ])
      # with a number_ob_unrolls of 30 and a unroll_length of 150 we get a tensor of size [4500, observation_shape]

      # # make unroll first
      # def unroll_first(unroll_data):
      #   unroll_data = unroll_data.swapaxes(0, 1)
      #   return unroll_data.reshape([unroll_data.shape[0], -1] + list(unroll_data.shape[3:]))
      # unroll_data = sd_map(unroll_first, unroll_data) # TODO: this applies the unroll_first to each eolement in the unroll data


      # update normalization statistics
      # self.agent.update_normalization(unroll_data.observation)

      # for _ in range(num_updates):
        
        # shuffle and batch the data
        # with T.no_grad():
        #   permutation = T.randperm(unroll_data.observation.shape[1], device=device)
        #   def shuffle_batch(data):
        #     data = data[:, permutation]
        #     data = data.reshape([data.shape[0], num_minibatches, -1] +
        #                         list(data.shape[2:]))
        #     return data.swapaxes(0, 1)
          # epoch_td = sd_map(shuffle_batch, unroll_data) # TODO: same as above with sd_map()

        # for minibatch_i in range(num_minibatches):
        #   # td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td) # TODO: same as above with sd_map()
        
      loss = self.agent.loss(unroll_data)

      self.optim.zero_grad()
      loss.backward()
      self.optim.step()

      duration = time.time() - t
      total_steps += num_epochs * num_steps
      sps = num_epochs * num_steps / duration
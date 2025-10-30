from copy import deepcopy

import torch as T

from torch import optim
from typing import Any, Callable, Dict, Optional

from unitree_robot.common.datastructure import UnrollData, MultiUnrollData
from unitree_robot.train.environments import MujocoEnv
from unitree_robot.train.agents import PPOAgent

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
        optimizer_fn=optim.AdamW,
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
        self.env = env
        self.observation_space_size = self.env.get_observation_size()
        self.action_space_size = self.env.get_action_size()
        self.reward_size = len(env.experiment)

        # -- create agent
        self.agent = PPOAgent(
            input_size=self.observation_space_size,
            output_size=self.action_space_size,
            network_hidden_size=network_hidden_size,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            device=device
        ).to(device=device)

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

    def train_unroll(self, unroll_length: int, seed: int):
        """Return step data over multple unrolls."""

        unroll = UnrollData.initialize_empty(
            unroll_length=unroll_length,
            observation_size=self.observation_space_size,
            action_size=self.action_space_size,
            reward_size=self.reward_size,
            device=self.device,
        )

        # env warmup
        self.env.reset(seed=seed)
        action = T.Tensor(self.env.action_space.sample()).to(dtype=T.float32, device=self.device)
        observation, _ = self.env.step(action=action)

        for j in range(unroll_length):

            # decide which action to take depending on the environment observation (robot state)
            logits, action = self.agent.get_logits_action(observation=observation)

            # take a step in the environment and get its return values like the local reward for taking that action.
            observation, rewards = self.env.step(action=action)

            raw_rewards, scaled_rewards = T.Tensor([*rewards.values()]).T

            unroll.observation[j, :] = observation[:]
            unroll.logits[j, :] = logits[:]
            unroll.action[j, :] = action[:]
            unroll.reward[j, :] = scaled_rewards[:]
            # unrolls.done[i,j] = done

        return observation, unroll

    def train(
        self,
        epochs: int = 4,
        num_unrolls: int = 5,
        unroll_length: int = 160,
        minibatch_size: int = 16, # the size of individual sequences extracted from a set of larger sequences whos length is given by unroll_length
        seed: int = 0,
    ):

        assert unroll_length % minibatch_size == 0, "unroll_length must be divisible by minibatch_size"
        num_minibatches = unroll_length // minibatch_size

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

        # num_steps = batch_size * num_minibatches * unroll_length
        # num_epochs = num_timesteps // (num_steps * eval_frequency)
        # num_unrolls = batch_size * num_minibatches // self.env.num_envs

        for e in range(epochs):

            print(f"epoch: {e}")

            unrolls = []
            for u in range(num_unrolls):
                observation, unroll_data = self.train_unroll(unroll_length=unroll_length, seed=seed)
                unrolls.append(unroll_data)
            multi_unroll_data = MultiUnrollData.from_multiple_unrolls(unrolls=unrolls)

            print(f"finished : {multi_unroll_data.observation.shape}")

            # print(f"reconfigured : {new_observation.shape}")

            # e.g. with a number_ob_unrolls of 30 and a unroll_length of 150 we get a tensor of size [4500, observation_shape]
            # # make unroll first
            # def unroll_first(unroll_data):
            #   unroll_data = unroll_data.swapaxes(0, 1)
            #   return unroll_data.reshape([unroll_data.shape[0], -1] + list(unroll_data.shape[3:]))
            # unroll_data = sd_map(unroll_first, unroll_data) # TODO: this applies the unroll_first to each eolement in the unroll data

            # update normalization statistics # TODO: unsure what this does
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

            # loss = self.agent.loss(unroll_data)
            #
            # self.optim.zero_grad()
            # loss.backward()
            # self.optim.step()


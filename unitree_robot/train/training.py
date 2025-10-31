from copy import deepcopy
from tqdm import tqdm
import torch as T
from torch import optim
from typing import Any, Callable, Dict, Optional
from torch.utils.data import DataLoader
from unitree_robot.common.datastructure import UnrollData, MultiUnrollDataset
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
            policy_output_size=self.action_space_size * 2, # *2 for logits
            value_output_size=self.reward_size,
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

        with T.no_grad():

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

            observation = observation.to(dtype=T.float32, device=self.device)

            unroll.observation[0, :] = observation[:]

            for j in range(1, unroll_length):

                # decide which action to take depending on the environment observation (robot state)
                logits, action = self.agent.sample_action(observation=observation)

                # take a step in the environment and get its return values like the local reward for taking that action.
                observation, rewards = self.env.step(action=action)
                observation = observation.to(dtype=T.float32, device=self.device)

                raw_rewards, scaled_rewards = T.Tensor([*rewards.values()]).to(dtype=T.float32, device=self.device).T

                unroll.observation[j, :] = observation[:]
                unroll.logits[j, :] = logits[:]
                unroll.action[j, :] = action[:]
                unroll.reward[j, :] = scaled_rewards[:]
                # unrolls.done[i,j] = done

            return observation, unroll

    def train(
        self,
        epochs: int = 4,
        train_batch_size: int = 16,
        num_unrolls: int = 4,
        unroll_length: int = 256,
        minibatch_size: int = 32, # the size of individual sequences extracted from a set of larger sequences whos length is given by unroll_length
        seed: int = 0,
    ):

        assert unroll_length % minibatch_size == 0, "unroll_length must be divisible by minibatch_size"
        num_minibatches = unroll_length // minibatch_size

        assert (num_unrolls * num_minibatches) % train_batch_size == 0, "(num_unrolls * num_minibatches) must be divisible by train_batch_size"

        for e in range(epochs):

            print(f"epoch: {e}")

            # Unroll a couple of times
            unrolls = []
            for u in tqdm(range(num_unrolls), desc="unrolling"):
                observation, unroll_data = self.train_unroll(unroll_length=unroll_length, seed=seed)
                unrolls.append(unroll_data)


            # convert the full sequences to sequence parts (minibatches)
            multi_unroll_dataset = MultiUnrollDataset(
                unrolls=unrolls,
                minibatch_size=minibatch_size,
                num_minibatches=num_minibatches,
                minibatched=True,
            )

            dataloader = DataLoader(
                multi_unroll_dataset,
                batch_size=train_batch_size,
                pin_memory_device=self.device,
            )

            loss_average = 0
            for i, batch in enumerate(dataloader):
                losses = self.agent(
                    observation=batch["observation"],
                    logits=batch["logits"],
                    action=batch["action"],
                    reward=batch["reward"],
                )

                # sum loss
                loss = losses["policy_loss"] + losses["value_loss"] + losses["entropy_loss"]
                loss_average += loss.detach().cpu().item()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            print(f"\tloss -> [ {loss_average / len(dataloader)} ]")


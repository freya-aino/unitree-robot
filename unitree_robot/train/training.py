import time
from tqdm import tqdm
import torch as T
import torch.nn as nn
from torch import optim
import mlflow
from torch.utils.data import DataLoader
from unitree_robot.common.datastructure import UnrollData, MultiUnrollDataset
from unitree_robot.train.environments import MujocoEnv
from unitree_robot.train.agents import PPOAgent
from unitree_robot.train.experiments import Experiment


class Trainer:
    def __init__(
        self,
        env: MujocoEnv,
        experiment: Experiment,
        device: str,
        network_hidden_size: int,
        network_layers: int,
        reward_scaling: float,
        lambda_: float,
        epsilon: float,
        discounting: float,
        learning_rate: float,
        max_gradient_norm: float,
        optimizer_fn=optim.AdamW,
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
        self.max_gradient_norm = max_gradient_norm

        # -- create agent
        self.agent = PPOAgent(
            input_size=self.observation_space_size,
            policy_output_size=self.action_space_size * 2,  # *2 for logits
            value_output_size=1,
            network_hidden_size=network_hidden_size,
            network_layers=network_layers,
            discounting=discounting,
            lambda_=lambda_,
            epsilon=epsilon,
            reward_scaling=reward_scaling,
            device=device,
        ).to(device=device)

        # -- set up optimizer
        self.optim = optimizer_fn(self.agent.parameters(), lr=learning_rate)

        # -- set experiment
        self.experiment = experiment

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

    def unroll(self, unroll_length: int, seed: int):
        """Return step data over multple unrolls."""

        unroll = UnrollData.initialize_empty(
            unroll_length=unroll_length,
            observation_size=self.observation_space_size,
            action_size=self.action_space_size,
        )

        # env warmup
        self.env.reset(seed=seed)
        action = T.Tensor(self.env.action_space.sample()).to(dtype=T.float32)
        last_observation, mj_data = self.env.step(action=action)
        last_observation = last_observation.to(dtype=T.float32, device=self.device)

        mean_rewards = {r: 0.0 for r in self.experiment(mj_data=mj_data)}

        for j in range(0, unroll_length):
            assert ~last_observation.isnan().any(), "NaN observation detected"
            assert ~last_observation.isinf().any(), "Inf observation detected"

            # decide which action to take depending on the environment observation (robot state)
            logits, action = self.agent.get_action(observation=last_observation)

            # squeeze the batch dimension and the sequence dimension
            action = action.cpu().squeeze().to(device="cpu")
            logits = logits.cpu().squeeze().to(device="cpu")

            # take a step in the environment and get its return values like the local reward for taking that action.
            observation, mj_data = self.env.step(action=action)
            rewards = self.experiment(mj_data=mj_data)
            scaled_reward_mean = T.Tensor([*rewards.values()]).mean()

            # log rewards
            for r in rewards:
                mean_rewards[r] += rewards[r] / unroll_length

            assert ~scaled_reward_mean.isnan().any(), "NaN reward detected"
            assert ~scaled_reward_mean.isinf().any(), "Inf reward detected"
            assert ~action.isnan().any(), "NaN action detected"
            assert ~action.isinf().any(), "Inf action detected"
            assert ~logits.isnan().any(), "NaN logits detected"
            assert ~logits.isinf().any(), "Inf logits detected"

            unroll.observation[j, :] = last_observation[:]
            unroll.logits[j, :] = logits[:]
            unroll.action[j, :] = action[:]
            unroll.reward[j, :] = scaled_reward_mean
            # unrolls.done[i,j] = done

            last_observation = observation.to(dtype=T.float32, device=self.device)

        return unroll, mean_rewards

    def train(
        self,
        epochs: int,
        train_batch_size: int,
        num_unrolls: int,
        unroll_length: int,
        minibatch_size: int,
        entropy_loss_scale: float,
        value_loss_scale: float,
        policy_loss_scale: float,
        seed: int,
    ):
        assert unroll_length % minibatch_size == 0, (
            "unroll_length must be divisible by minibatch_size"
        )
        num_minibatches = unroll_length // minibatch_size

        assert (num_unrolls * num_minibatches) % train_batch_size == 0, (
            "(num_unrolls * num_minibatches) must be divisible by train_batch_size"
        )

        for e in tqdm(range(epochs), "training"):
            # --- UNROLL

            time_to_unroll = time.perf_counter()

            # Unroll a couple of times
            self.agent.eval()
            with T.no_grad():
                unrolls = []
                mean_rewards = {}
                for _ in range(num_unrolls):
                    unroll, mr = self.unroll(unroll_length=unroll_length, seed=seed)
                    unrolls.append(unroll)
                    for k in mr:
                        if k in mean_rewards.keys():
                            mean_rewards[k] += mr[k] / num_unrolls
                        else:
                            mean_rewards[k] = mr[k] / num_unrolls

            for k in mean_rewards:
                mlflow.log_metric(k, mean_rewards[k], step=e)

            mlflow.log_metric(
                "mean_time_per_unroll_step_ms",
                (time.perf_counter() - time_to_unroll)
                / (num_unrolls * unroll_length)
                * 1000,
                step=e,
            )

            # --- CREATE DATASET

            # convert the full sequences to sequence parts (minibatches)
            multi_unroll_dataset = MultiUnrollDataset(
                unrolls=unrolls,
                minibatch_size=minibatch_size,
                num_minibatches=num_minibatches,
                minibatched=True,
            )

            assert ~multi_unroll_dataset.action.isnan().any(), (
                "Action contains NaN values"
            )
            assert ~multi_unroll_dataset.action.isinf().any(), (
                "Action contains infinite values"
            )
            assert ~multi_unroll_dataset.reward.isnan().any(), (
                "Reward contains NaN values"
            )
            assert ~multi_unroll_dataset.reward.isinf().any(), (
                "Reward contains infinite values"
            )
            assert ~multi_unroll_dataset.logits.isnan().any(), (
                "Logits contains NaN values"
            )
            assert ~multi_unroll_dataset.logits.isinf().any(), (
                "Logits contains infinite values"
            )
            assert ~multi_unroll_dataset.observations.isnan().any(), (
                "Observations contain NaN values"
            )
            assert ~multi_unroll_dataset.observations.isinf().any(), (
                "Observations contain infinite values"
            )

            mlflow.log_metric(
                "mean_reward",
                multi_unroll_dataset.reward.mean().detach().cpu().item(),
                step=e,
            )

            dataloader = DataLoader(
                multi_unroll_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                pin_memory=True,
            )

            # --- TRAINING

            self.agent.train()

            loss_averages = {
                "policy_loss": 0,
                "value_loss": 0,
                "entropy_loss": 0,
            }
            for _, batch in enumerate(dataloader):
                observations = batch["observations"].to(self.device)
                logits = batch["logits"].to(self.device)
                actions = batch["actions"].to(self.device)
                rewards = batch["rewards"].to(self.device)

                losses = self.agent(
                    observations=observations,
                    logits=logits,
                    actions=actions,
                    rewards=rewards,
                )

                # logging
                loss_averages["policy_loss"] += (
                    losses["policy_loss"].detach().cpu().item()
                )
                loss_averages["value_loss"] += (
                    losses["value_loss"].detach().cpu().item()
                )
                loss_averages["entropy_loss"] += (
                    losses["entropy_loss"].detach().cpu().item()
                )

                policy_loss = losses["policy_loss"] * policy_loss_scale
                value_loss = losses["value_loss"] * value_loss_scale
                entropy_loss = losses["entropy_loss"] * entropy_loss_scale

                assert ~policy_loss.isnan(), f"policy_loss is NaN"
                assert ~policy_loss.isinf(), f"policy_loss is Inf"
                assert ~value_loss.isnan(), f"value_loss is NaN"
                assert ~value_loss.isinf(), f"value_loss is Inf"
                assert ~entropy_loss.isnan(), f"entropy_loss is NaN"
                assert ~entropy_loss.isinf(), f"entropy_loss is Inf"

                # sum loss and backpropagate
                loss = policy_loss + value_loss + entropy_loss
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), max_norm=self.max_gradient_norm
                )
                self.optim.step()

            for k in loss_averages:
                mlflow.log_metric(k, loss_averages[k], step=e)

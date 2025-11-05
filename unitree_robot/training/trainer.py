import time
from tqdm import tqdm
import torch as T
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader
from unitree_robot.common.datastructure import UnrollData, MultiUnrollDataset
from unitree_robot.training.environments import MujocoEnv
from unitree_robot.training.experiments import Experiment
from unitree_robot.common.agents import PPOAgent


class Trainer:
    def __init__(
        self,
        environment: MujocoEnv,
        experiment: Experiment,
        agent: PPOAgent,
        device: T.device,
        max_gradient_norm: float,
        seed: int,
        optimizer,
    ) -> None:
        # -- variables
        self.environment = environment
        self.experiment = experiment
        self.agent = agent.to(device)
        self.device = device
        self.seed = seed

        # -- set up and grad norm
        self.optimizer = optimizer
        self.max_gradient_norm = max_gradient_norm

    def unroll_step(self, observation: T.Tensor):
        # assert ~last_observation.isnan().any(), "NaN observation detected"
        # assert ~last_observation.isinf().any(), "Inf observation detected"

        # decide which action to take depending on the environment observation (robot state)
        logits, action = self.agent.get_action(observation=observation)

        # squeeze the batch dimension and the sequence dimension
        action = action.cpu().squeeze().to(device="cpu")
        logits = logits.cpu().squeeze().to(device="cpu")

        # take a step in the environment and get its return values like the local reward for taking that action.
        new_observation, mj_data = self.environment.step(action=action)
        rewards = self.experiment(mj_data=mj_data)
        scaled_reward_mean = T.Tensor([*rewards.values()]).mean()

        # assert ~scaled_reward_mean.isnan().any(), "NaN reward detected"
        # assert ~scaled_reward_mean.isinf().any(), "Inf reward detected"
        # assert ~action.isnan().any(), "NaN action detected"
        # assert ~action.isinf().any(), "Inf action detected"
        # assert ~logits.isnan().any(), "NaN logits detected"
        # assert ~logits.isinf().any(), "Inf logits detected"

        # for n in self.env.actuator_names:
        #     value = self.env.data.actuator(n).velocity[0]
        #     mlflow.log_metric(f"actuator velocity - {n}", value, step=j)

        return {
            "observation": new_observation,
            "logits": logits,
            "action": action,
            "reward": scaled_reward_mean,
            "raw_rewards": rewards,
        }

    def unroll(self, unroll_length: int, seed: int):
        """Return step data over multple unrolls."""

        unroll_data = UnrollData.initialize_empty(
            unroll_length=unroll_length,
            observation_size=self.environment.observation_space_size,
            action_size=self.environment.action_space_size,
        )

        # env warmup
        _ = self.environment.reset(seed=seed)
        action = T.Tensor(self.environment.action_space.sample()).to(dtype=T.float32)
        last_observation, mj_data = self.environment.step(action=action)
        last_observation = last_observation.to(dtype=T.float32)

        mean_rewards = {r: 0.0 for r in self.experiment(mj_data=mj_data)}
        for j in range(0, unroll_length):
            step_out = self.unroll_step(
                observation=last_observation.to(device=self.device)
            )

            # log raw rewards
            for r in step_out["raw_rewards"]:
                mean_rewards[r] += step_out["raw_rewards"][r] / unroll_length

            # track unroll data
            unroll_data.observation[j, :] = last_observation[:]
            unroll_data.logits[j, :] = step_out["logits"][:]
            unroll_data.actions[j, :] = step_out["action"][:]
            unroll_data.rewards[j, :] = step_out["reward"]

            # set last observation
            last_observation = step_out["observation"]

        return unroll_data, mean_rewards

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
                    unroll, mr = self.unroll(
                        unroll_length=unroll_length, seed=self.seed
                    )
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
            multi_unroll_dataset.validate()

            mlflow.log_metric(
                "mean_reward",
                multi_unroll_dataset.rewards.mean().detach().cpu().item(),
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
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), max_norm=self.max_gradient_norm
                )
                self.optimizer.step()

            for k in loss_averages:
                mlflow.log_metric(k, loss_averages[k], step=e)

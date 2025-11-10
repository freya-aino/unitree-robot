import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from unitree_robot.common.datastructure import UnrollData
from unitree_robot.common.networks import BasicPolicyValueNetwork


class PPOAgent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        network_hidden_size: int,
        num_hidden_layers: int,
        discounting: float,
        lambda_: float,
        epsilon: float,
        moving_average_window_size: int,
        reward_scaling: float,
        train_sequence_length: int,
        policy_loss_scale: float,
        value_loss_scale: float,
        entropy_loss_scale: float,
        softplus_sharpness_factor: float,
    ):
        super().__init__()

        self.network = BasicPolicyValueNetwork(
            input_size=observation_size,
            policy_output_size=action_size * 2,
            value_output_size=1,
            num_hidden_layers=num_hidden_layers,
            hidden_size=network_hidden_size,
        )

        self.moving_average_window_size = moving_average_window_size
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.epsilon = epsilon
        self.softplus_sharpness_factor = softplus_sharpness_factor

        self.policy_loss_scale = policy_loss_scale
        self.value_loss_scale = value_loss_scale
        self.entropy_loss_scale = entropy_loss_scale

        # decay matrix
        decay_factor = self.discounting * lambda_
        self.decay_factor_matrix = Parameter(
            T.triu(
                decay_factor
                ** T.clamp(
                    T.arange(train_sequence_length - 1).unsqueeze(0)
                    - T.arange(train_sequence_length - 1).unsqueeze(1),
                    min=0,
                )
            ).unsqueeze(0),
            requires_grad=False,
        )

    def create_distribution(self, logits: T.Tensor):
        loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
        return Normal(
            loc=loc,
            scale=F.softplus(scale, beta=self.softplus_sharpness_factor) + 0.001,
        )
        # return Normal(
        #     loc=loc,
        #     scale=0.001 # TODO for simulating a real world discrete control signal
        # )

    def get_action_and_logits(self, observation: T.Tensor):
        # observation = self.normalize(observation # TODO - this is still missing from the original code
        logits = self.network.policy_forward(observation)

        dist = self.create_distribution(logits)
        action = dist.rsample()
        return action, logits

    @staticmethod
    def jacobian_entropy(dist: Normal):
        entropy = dist.entropy()
        sample = dist.rsample()
        jacobian = 2 * (math.log(2) - sample - F.softplus(-2 * sample))
        return (entropy + jacobian).sum(-1)

    @staticmethod
    def jacobian_log_prob(dist: Normal, sample: T.Tensor):
        log_p = dist.log_prob(sample)
        jacobian = 2 * (math.log(2) - sample - F.softplus(-2 * sample))
        return (log_p - jacobian).sum(dim=-1).mean(dim=0)

    def train_step(self, unroll_data: UnrollData):
        observations = unroll_data.observations
        logits = unroll_data.logits
        actions = unroll_data.actions
        rewards = unroll_data.rewards

        # normalize rewards
        std, mu = T.std_mean(rewards, dim=1, keepdim=True)
        rewards = (rewards - mu) / std

        # moving average rewards
        rewards = F.avg_pool1d(
            rewards.transpose(1, 2),
            stride=1,
            kernel_size=self.moving_average_window_size,
            padding=self.moving_average_window_size // 2,
        ).transpose(1, 2)[:, :-1, :]

        policy_logits = self.network.policy_forward(observations)
        values = self.network.value_forward(observations)

        # format the individual sequences
        values_t_1 = values[:, 1:]
        bootstrap_value = values[:, -1:]
        values_t = values[:, :-1]
        policy_logits = policy_logits[:, :-1]
        rewards = rewards[:, :-1]
        actions = actions[:, :-1]
        logits = logits[:, :-1]

        # compute GAE
        with T.no_grad():
            # calculate deltas for each timestep simultainously
            deltas = rewards + self.discounting * values_t_1 - values_t

            # calculate discounted deltas
            discounted_deltas = T.matmul(self.decay_factor_matrix, deltas)

            vs_t = discounted_deltas + values_t
            vs_t_1 = T.cat([vs_t[:, 1:], bootstrap_value], 1)
            advantages = rewards + self.discounting * vs_t_1 - values_t

        behaviour_dist = self.create_distribution(logits)
        behaviour_action_log_probs = self.jacobian_log_prob(behaviour_dist, actions)
        policy_dist = self.create_distribution(policy_logits)
        policy_action_log_probs = self.jacobian_log_prob(policy_dist, actions)

        rho_s = T.exp(policy_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -T.mean(T.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        value_loss = F.smooth_l1_loss(vs_t, values_t)

        # Entropy loss
        entropy_loss = -self.jacobian_entropy(policy_dist).mean()

        return (
            policy_loss * self.policy_loss_scale
            + value_loss * self.value_loss_scale
            + entropy_loss * self.entropy_loss_scale,
        {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        })

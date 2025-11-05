import math
from mpmath import polar
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from unitree_robot.common.networks import BasicPolicyValueNetwork
# from unitree_robot.common.datastructure import UnrollData


class PPOAgent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        input_size: int,
        policy_output_size: int,
        value_output_size: int,
        network_hidden_size: int,
        network_layers: int,
        discounting: float,
        reward_scaling: float,
        device: str,
    ):
        super().__init__()

        self.network = BasicPolicyValueNetwork(
            input_size=input_size,
            network_layers=network_layers,
            policy_output_size=policy_output_size,
            value_output_size=value_output_size,
            hidden_size=network_hidden_size,
        )

        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.3

        self.device = device

    def create_distribution(self, logits: T.Tensor):
        loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
        return Normal(
            loc=loc,
            scale=F.softplus(scale),
        )
        # return Normal(
        #     loc=loc,
        #     scale=0.001 # TODO for simulating a real world discrete control signal
        # )

    def get_action(self, observation: T.Tensor):
        # observation = self.normalize(observation
        assert ~observation.isnan().any(), "Observation contains NaN values"
        assert ~observation.isinf().any(), "Observation contains infinite values"

        if self.eval:
            assert len(observation.shape) == 1, (
                f"Expected 1D tensor when evaluating, got {observation.shape}"
            )
            observation = observation.unsqueeze(0).unsqueeze(0)

        logits = self.network.policy_forward(observation)

        # print(logits)

        assert ~logits.isnan().any(), "Logits contain NaN values"
        assert ~logits.isinf().any(), "Logits contain infinite values"

        dist = self.create_distribution(logits)
        action = dist.sample()
        return logits, action

    def forward(
        self,
        observations: T.Tensor,
        logits: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
        moving_average_window_size: int = 16
    ):

        # moving average rewards
        rewards = F.avg_pool1d(
            rewards.transpose(1, 2), stride=1, kernel_size=moving_average_window_size, padding=moving_average_window_size//2
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

        # print("rewards", rewards.min(), rewards.max())
        # print("values", values.min(), values.max())

        # compute GAE
        with T.no_grad():
            # calculate deltas from values to rewards
            # basically a value from how the current reward + the value from the next step forward
            # compares to the expected value from this step forward
            # (you can call this a surprise value)
            # larger than 0 means its better than expected
            # less than 0 means its worse than expected
            deltas = rewards + self.discounting * values_t_1 - values_t

            # print("deltas:", deltas.min(), deltas.max(), deltas.isnan().any())

            # calculate discounted deltas
            powers = (
                T.arange(deltas.shape[1]).unsqueeze(0)
                - T.arange(deltas.shape[1]).unsqueeze(1)
            ).to(device=self.device)
            decay_factor = (self.lambda_ * self.discounting) ** T.clamp(powers, min=0)
            discount_matrix = T.triu(decay_factor).unsqueeze(0)
            discounted_deltas = T.matmul(discount_matrix, deltas)

            # print(
            #     "discounted deltas:",
            #     discounted_deltas.min(),
            #     discounted_deltas.max(),
            #     discounted_deltas.isnan().any(),
            # )

            vs_t = discounted_deltas + values_t
            vs_t_1 = T.cat([vs_t[:, 1:], bootstrap_value], 1)
            advantages = rewards + self.discounting * vs_t_1 - values_t

        behaviour_dist = self.create_distribution(logits)
        behaviour_action_log_probs = behaviour_dist.log_prob(actions)
        policy_dist = self.create_distribution(policy_logits)
        policy_action_log_probs = policy_dist.log_prob(actions)

        # print(
        #     "policy dist",
        #     policy_dist.loc.min(),
        #     policy_dist.loc.max(),
        #     policy_dist.scale.min(),
        #     policy_dist.scale.max(),
        # )
        # print("behaviour action", behaviour_action_log_probs.mean(), policy_action_log_probs.mean())
        # print("vs_t", vs_t.min(), vs_t.max())

        rho_s = T.exp(policy_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -T.mean(T.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        value_loss = F.smooth_l1_loss(vs_t, values_t)

        # Entropy loss
        entropy_loss = -policy_dist.entropy().mean()

        # print(f"policy loss: {policy_loss}")
        # print(f"value loss: {value_loss}")
        # print(f"entropy loss: {entropy_loss}")

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        }

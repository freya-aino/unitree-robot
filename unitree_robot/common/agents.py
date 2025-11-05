import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from unitree_robot.common.networks import BasicPolicyValueNetwork


class PPOAgent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        input_size: int,
        policy_output_size: int,
        value_output_size: int,
        network_hidden_size: int,
        num_hidden_layers: int,
        discounting: float,
        lambda_: float,
        epsilon: float,
        moving_average_window_size: int,
        reward_scaling: float,
        train_sequence_length: int,
    ):
        super().__init__()

        self.network = BasicPolicyValueNetwork(
            input_size=input_size,
            num_hidden_layers=num_hidden_layers,
            policy_output_size=policy_output_size,
            value_output_size=value_output_size,
            hidden_size=network_hidden_size,
        )

        self.moving_average_window_size = moving_average_window_size
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.epsilon = epsilon

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
            scale=F.softplus(scale) + 0.001,
        )
        # return Normal(
        #     loc=loc,
        #     scale=0.001 # TODO for simulating a real world discrete control signal
        # )

    def get_action(self, observation: T.Tensor):
        # observation = self.normalize(observation # TODO - this is still missing from the original code
        if self.eval:
            assert len(observation.shape) == 1, (
                f"Expected 1D tensor when evaluating, got {observation.shape}"
            )
            observation = observation.unsqueeze(0).unsqueeze(0)

        logits = self.network.policy_forward(observation)

        dist = self.create_distribution(logits)
        action = dist.sample()
        return logits, action

    def forward(
        self,
        observations: T.Tensor,
        logits: T.Tensor,
        actions: T.Tensor,
        rewards: T.Tensor,
    ):
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
            deltas = rewards + self.discounting * values_t_1 - values_t

            # calculate discounted deltas

            discounted_deltas = T.matmul(self.decay_factor_matrix, deltas)

            vs_t = discounted_deltas + values_t
            vs_t_1 = T.cat([vs_t[:, 1:], bootstrap_value], 1)
            advantages = rewards + self.discounting * vs_t_1 - values_t

        behaviour_dist = self.create_distribution(logits)
        behaviour_action_log_probs = behaviour_dist.log_prob(actions)
        policy_dist = self.create_distribution(policy_logits)
        policy_action_log_probs = policy_dist.log_prob(actions)

        rho_s = T.exp(policy_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -T.mean(T.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        value_loss = F.smooth_l1_loss(vs_t, values_t)

        # Entropy loss
        entropy_loss = -policy_dist.entropy().mean()

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        }

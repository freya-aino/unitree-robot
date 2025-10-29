import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from unitree_robot.common.networks import BasicPolicyValueNetwork
from unitree_robot.common.datastructure import UnrollData


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
        super().__init__()

        self.network = BasicPolicyValueNetwork(input_size, output_size, network_hidden_size)

        self.num_steps = T.zeros((), device=device)
        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.3
        self.device = device

    def dist_create(self, logits):
        """Normal followed by tanh.

        T.distribution doesn't work with T.jit, so we roll our own."""
        loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + .001
        return loc, scale

    def dist_sample_no_postprocess(self, loc, scale):
        return T.normal(loc, scale)

    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + T.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * T.ones_like(loc)
        dist = T.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + T.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    # def update_normalization(self, observation):
    #   self.num_steps += observation.shape[0] * observation.shape[1]
    #   input_to_old_mean = observation - self.running_mean
    #   mean_diff = T.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
    #   self.running_mean = self.running_mean + mean_diff
    #   input_to_new_mean = observation - self.running_mean
    #   var_diff = T.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
    #   self.running_variance = self.running_variance + var_diff

    # def normalize(self, observation):
    #   variance = self.running_variance / (self.num_steps + 1.0)
    #   variance = T.clip(variance, 1e-6, 1e6)
    #   return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    def get_logits_action(self, observation):
        # observation = self.normalize(observation)
        logits = self.network.policy_forward(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

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

    def loss(self, unroll_data: UnrollData):
        # observation = self.normalize(unroll_data.observation)
        observation = unroll_data.observation
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


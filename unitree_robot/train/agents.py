import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from unitree_robot.common.networks import BasicPolicyValueNetwork
from unitree_robot.common.datastructure import UnrollData


class PPOAgent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        input_size: int,
        policy_output_size: int,
        value_output_size: int,
        network_hidden_size: int,
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
        device: str
    ):
        super().__init__()

        self.network = BasicPolicyValueNetwork(
            input_size,
            policy_output_size,
            value_output_size,
            network_hidden_size
        )

        self.num_steps = T.zeros((), device=device)
        # self.entropy_cost = nn.Parameter(T.Tensor([entropy_cost]), requires_grad=False)
        # self.discounting = nn.Parameter(T.Tensor([discounting]), requires_grad=False)
        # self.reward_scaling = nn.Parameter(T.Tensor([reward_scaling]), requires_grad=False)
        # self.lambda_ = nn.Parameter(T.Tensor([0.95]), requires_grad=False)
        # self.epsilon = nn.Parameter(T.Tensor([0.3]), requires_grad=False)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.3

        self.device = device

    def create_distribution(self, logits: T.Tensor):
        """T.distribution doesn't work with T.jit, so we roll our own."""
        loc, scale = T.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + .001
        return loc, scale
        # _, _, logit_size = logits.shape
        # return Normal(
        #     loc=logits[..., logit_size//2:],
        #     scale=F.softplus(logits[..., :logit_size//2]) + .001,
        # )

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

    def dist_log_prob(
        self,
        loc: T.Tensor,
        scale: T.Tensor,
        dist: T.Tensor
    ):
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

    def sample_action(self, observation: T.Tensor):
        # observation = self.normalize(observation)
        logits = self.network.policy_forward(observation)
        loc, scale = self.create_distribution(logits)
        action = Normal(loc, scale).sample()
        return logits, action

    def compute_gae(
        self,
        rewards: T.Tensor,
        values: T.Tensor,
        bootstrap_value: T.Tensor
    ):
        batch_size, sequence_length, _ = values.shape

        bootstrap_value = bootstrap_value.unsqueeze(dim=1)

        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = T.cat([values[:, 1:], bootstrap_value], dim=1)

        # calculate deltas from values to rewards
        discounted_value = self.discounting * values_t_plus_1
        deltas = rewards + discounted_value - values

        # calculate discounted deltas
        powers = (T.arange(sequence_length).unsqueeze(0) - T.arange(sequence_length).unsqueeze(1)).to(device=self.device)
        decay_factor = self.lambda_ * self.discounting # TODO : this is unnesesarily split into two
        discount_matrix = T.triu(decay_factor ** T.clamp(powers, min=0)).unsqueeze(0)
        vs_minus_v_xs = T.matmul(discount_matrix, deltas)

        # acc = T.zeros_like(bootstrap_value)
        # vs_minus_v_xs = T.zeros_like(values)
        #
        # sequence_length = values.shape[1]
        # for ti in range(sequence_length-1, -1, -1):
        #     # ti = sequence_length - ti - 1
        #     acc = deltas[:, ti] + self.discounting * self.lambda_ * acc
        #     vs_minus_v_xs[:, ti] = acc

        # print(f"bootstrap value shape: {bootstrap_value.shape}")
        # print(f"values shape: {values.shape}")
        # print(f"rewards shape: {rewards.shape}")
        # print(f"values_t_plus_1 shape: {values_t_plus_1.shape}")
        # print(f"deltas shape: {deltas.shape}")
        # print(f"vs_minus_v_xs shape: {vs_minus_v_xs.shape}")

        # Add V(x_s) to get v_s
        vs = vs_minus_v_xs + values

        vs_t_plus_1 = T.cat([vs[:, 1:], bootstrap_value], 1)
        advantages = rewards + self.discounting * vs_t_plus_1 - values
        return vs, advantages

    # def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
    #     truncation_mask = 1 - truncation
    #     # Append bootstrapped value to get [v1, ..., v_t+1]
    #     values_t_plus_1 = T.cat([values[1:], T.unsqueeze(bootstrap_value, 0)], dim=0)
    #     deltas = reward + self.discounting * (
    #             1 - termination) * values_t_plus_1 - values
    #     deltas *= truncation_mask
    #
    #     acc = T.zeros_like(bootstrap_value)
    #     vs_minus_v_xs = T.zeros_like(truncation_mask)
    #
    #     for ti in range(truncation_mask.shape[0]):
    #         ti = truncation_mask.shape[0] - ti - 1
    #         acc = deltas[ti] + self.discounting * (
    #                 1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
    #         vs_minus_v_xs[ti] = acc
    #
    #     # Add V(x_s) to get v_s.
    #     vs = vs_minus_v_xs + values
    #     vs_t_plus_1 = T.cat([vs[1:], T.unsqueeze(bootstrap_value, 0)], 0)
    #     advantages = (reward + self.discounting *
    #                   (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    #     return vs, advantages


    def forward(
        self,
        observation: T.Tensor,
        logits: T.Tensor,
        action: T.Tensor,
        reward: T.Tensor,
    ):
        # throw out the first action, reward and logits because they are empty
        action = action[:, 1:]
        reward = reward[:, 1:] * self.reward_scaling
        logits = logits[:, 1:]

        # termination = done * (1 - truncation) # TODO: both done and truncation are not used currently

        # observation = self.normalize(unroll_data.observation)
        # policy_logits = self.network.policy_forward(observation[:-1])
        # baseline = self.network.value_forward(observation)
        # baseline = T.squeeze(baseline, dim=-1)
        policy_logits = self.network.policy_forward(observation[:, :-1])
        value_baseline = self.network.value_forward(observation)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = value_baseline[:, -1]
        value_baseline = value_baseline[:, :-1]

        original_loc, original_scale = self.create_distribution(logits)
        policy_loc, policy_scale = self.create_distribution(policy_logits)

        behaviour_action_log_probs = self.dist_log_prob(original_loc, original_scale, action)
        target_action_log_probs = self.dist_log_prob(policy_loc, policy_scale, action)

        # print(f"behaviour_action_log_probs shape: {behaviour_action_log_probs.shape}")
        # print(f"target_action_log_probs shape: {target_action_log_probs.shape}")

        with T.no_grad():
            vs, advantages = self.compute_gae(
                # truncation=unroll_data.truncation,
                # termination=termination,
                rewards=reward,
                values=value_baseline,
                bootstrap_value=bootstrap_value
            )

        advantages = advantages.mean(dim=-1)
        # vs = vs.mean(dim=-1)

        # print(f"vs shape", vs.shape)
        # print("advantages shape", advantages.shape)

        rho_s = T.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -T.mean(T.minimum(surrogate_loss1, surrogate_loss2))


        # print(f"rho_s shape: {rho_s.shape}")
        # print(f"policy_loss: {policy_loss}")

        # Value function loss
        # v_error = vs - value_baseline
        # v_loss = (v_error).pow(2).mean()
        v_loss = F.smooth_l1_loss(vs, value_baseline)

        # Entropy reward
        entropy = T.mean(self.dist_entropy(policy_loc, policy_scale))
        entropy_loss = self.entropy_cost * -entropy

        return {
            "policy_loss": policy_loss,
            "value_loss": v_loss,
            "entropy_loss": entropy_loss
        }


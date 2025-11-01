import torch as T
import torch.nn as nn
from typing import Tuple

class NetworkBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, act_f):
        super().__init__()

        self.nn1 = nn.Linear(input_size, output_size)
        self.ln = nn.LayerNorm(output_size)
        self.act = act_f()
        self.nn2 = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x: T.Tensor) -> T.Tensor:
        a = self.act(self.ln(self.nn1(x)))
        b = self.act(self.bn(self.nn2(x).permute(0, 2, 1)).permute(0, 2, 1))
        return (a + b) / 2

class BasicPolicyValueNetwork(nn.Module):
    
    def __init__(self, input_size: int, network_layers: int, policy_output_size: int, value_output_size: int, hidden_size: int):

        super().__init__()

        hidden_sizes = [hidden_size] * network_layers

        self.policy_network = nn.Sequential(
            NetworkBlock(input_size, hidden_size, nn.ELU),
            *[NetworkBlock(hidden_size, hidden_size, nn.ELU) for _ in range(network_layers)],
            NetworkBlock(hidden_size, policy_output_size, nn.Tanh)
        )

        self.value_network = nn.Sequential(
            NetworkBlock(input_size, hidden_size, nn.ELU),
            *[NetworkBlock(hidden_size, hidden_size, nn.ELU) for _ in range(network_layers)],
            NetworkBlock(hidden_size, policy_output_size, nn.ELU)
        )

    def forward(self, observation: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        return (
            self.policy_network(observation), # "policy_logits"
            self.value_network(observation) # "baseline"
        )


    def policy_forward(self, x: T.Tensor):
        return self.policy_network(x)

    def value_forward(self, x: T.Tensor):
        return self.value_network(x)

from multiprocessing import Value
import torch as T
import torch.nn as nn
from typing import Tuple


class NetworkBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, act_f):
        super().__init__()

        # self.bn = nn.BatchNorm1d(input_size)
        # self.ln = nn.LayerNorm(input_size)
        #
        # self.nn1 = nn.Linear(input_size, input_size)
        # self.nn2 = nn.Linear(input_size, input_size)
        self.nn3 = nn.Linear(input_size, output_size)

        self.act = act_f()

    def forward(self, x: T.Tensor) -> T.Tensor:
        # x = self.act(self.ln(self.nn1(x)))
        # x = self.act(self.bn(self.nn2(x).permute(0, 2, 1)).permute(0, 2, 1))
        return self.act(self.nn3(x))


class BasicPolicyValueNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        network_layers: int,
        policy_output_size: int,
        value_output_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.policy_network = nn.Sequential(
            NetworkBlock(input_size, hidden_size, nn.ELU),
            *[
                NetworkBlock(hidden_size, hidden_size, nn.ELU)
                for _ in range(network_layers)
            ],
            NetworkBlock(hidden_size, policy_output_size, nn.Identity),
        )

        self.value_network = nn.Sequential(
            NetworkBlock(input_size, hidden_size, nn.ELU),
            *[
                NetworkBlock(hidden_size, hidden_size, nn.ELU)
                for _ in range(network_layers)
            ],
            NetworkBlock(hidden_size, value_output_size, nn.ReLU),
        )

    def policy_forward(self, x: T.Tensor) -> T.Tensor:
        return self.policy_network(x)

    def value_forward(self, x: T.Tensor) -> T.Tensor:
        return self.value_network(x)

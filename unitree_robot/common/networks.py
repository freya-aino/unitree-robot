import torch as T
import torch.nn as nn
from typing import Tuple

class BasicPolicyValueNetwork(nn.Module):
    
    def __init__(self, input_size: int, policy_output_size: int, value_output_size: int, hidden_size: int):

        super().__init__()

        self.policy_network = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ELU(), 
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, policy_output_size), # * 2 because it should output logits
            nn.ELU(), 
        )

        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ELU(), 
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, value_output_size),
            nn.ELU(),
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

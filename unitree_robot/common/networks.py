import torch as T
import torch.nn as nn

class BasicPolicyValueNetwork(nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int):

        self.policy_network = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ELU(), 
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size), 
            nn.ELU(), 
        )

        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.ELU(), 
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size), 
            nn.ELU(),
        )

    def policy_forward(self, x: T.Tensor):
        return(self.policy_network(x))

    def value_forward(self, x: T.Tensor):
        return(self.value_network(x))


import torch
import torch.nn as nn
    
class FiniteXEmbedding(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim, act=nn.GELU()):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.ModuleList()
        for i in range(num_layers):
            self.mlp.append(
                nn.Sequential(
                    act,
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for i in range(self.num_layers):
            x = self.mlp[i](x)
        x = self.output_layer(x)
        return x
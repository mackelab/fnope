import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, domain_dim, hidden_dim, output_dim,act=nn.GELU()):
        super().__init__()
        self.domain_dim = domain_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pos):
        if pos.ndim == 3:
            assert pos.shape[-1] == self.domain_dim, "pos should have shape (batch, npoints, ndim)"
        elif pos.ndim == 2:
            assert self.domain_dim == 1, "pos should have shape (batch, npoints, ndim) or (batch, npoints)"
            pos = pos.unsqueeze(-1) # (batch, npoints) -> (batch, npoints, 1)
        pemb = self.mlp(pos) # (batch, npoints, output_dim)

        
        return pemb
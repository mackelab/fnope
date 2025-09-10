import torch
import torch.nn as nn
from torch import Tensor
import math


class TimestepEmbedding(nn.Module):
    def __init__(self, num_freqs, hidden_dim, output_dim, pos_dim=1, act=nn.GELU()):
        super().__init__()
        self.num_freqs = num_freqs
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim

        self.register_buffer("freqs", torch.arange(1, num_freqs + 1) * math.pi)

        self.mlp = nn.Sequential(
            nn.Linear(2*num_freqs, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t):
        temb = self.freqs * t[..., None]
        temb = torch.cat((temb.cos(), temb.sin()), dim=-1)
        temb = self.mlp(temb)
        return temb
    


class SinusoidalTimeEmbedding(nn.Module):
    #adapted from https://github.com/sbi-dev/sbi/blob/merge_flow_builders_to_current_main/sbi/neural_nets/net_builders/vector_field_nets.py#L318
    """Sinusoidal time embedding as used in Vaswani et al. (2017).

    Can be used for time embedding in both vector field nets.
    """

    def __init__(self, embed_dim: int = 16, max_freq: float = 0.01):
        """Initialize sinusoidal embedding.

        args:
            embed_dim: dimension of the embedding (must be even)
            max_freq: maximum frequency denominator
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embedding dimension must be even")

        self.embed_dim = embed_dim
        self.max_freq = max_freq
        # compute frequency bands
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(max_freq) / embed_dim)
        )
        self.register_buffer("div_term", div_term)
        self.out_features = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed time using transformer sinusoidal embeddings.

        args:
            t: time tensor of shape (batch_size, 1) or (batch_size,) or scalar ()

        returns:
            time embedding of shape (batch_size, embed_dim) or (embed_dim,)
            for scalar input
        """
        # handle scalar inputs (0-dim tensors)
        if t.ndim == 0:
            # create output for a single time point
            time_embedding = torch.zeros(self.embed_dim, device=t.device)
            time_embedding[0::2] = torch.sin(t * self.div_term)
            time_embedding[1::2] = torch.cos(t * self.div_term)
            return time_embedding.unsqueeze(0)

        # ensure time has the right shape for broadcasting
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        # create embeddings pe(pos, 2i) = sin(pos/1000^(2i/d))
        # pe(pos, 2i+1) = cos(pos/1000^(2i/d))
        time_embedding = torch.zeros(t.shape[:-1] + (self.embed_dim,), device=t.device)
        time_embedding[:, 0::2] = torch.sin(t * self.div_term)
        time_embedding[:, 1::2] = torch.cos(t * self.div_term)

        return time_embedding


class RandomFourierTimeEmbedding(nn.Module):
    #adapted from https://github.com/sbi-dev/sbi/blob/merge_flow_builders_to_current_main/sbi/neural_nets/net_builders/vector_field_nets.py#L318
    """Gaussian random features for encoding time steps.

    This is to be used as a utility for score-matching."""

    def __init__(self, embed_dim=256, scale=30.0, learnable=True):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.embed_dim = embed_dim
        self.scale = scale
        if not learnable:
            self.register_buffer("W", torch.randn(embed_dim // 2) * scale)
        else:
            self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale)

    def forward(self, times: Tensor):
        times_proj = times[:, None] * self.W[None, :] * 2 * torch.pi
        embedding = torch.cat([torch.sin(times_proj), torch.cos(times_proj)], dim=-1)
        return torch.squeeze(embedding, dim=1)
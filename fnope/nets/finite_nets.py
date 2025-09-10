import torch
from torch import nn,Tensor
from typing import Union, Tuple, Optional, Callable

from abc import ABC, abstractmethod

# Flow matching nets adapted from 
# https://github.com/sbi-dev/sbi/tree/main/sbi/neural_nets/net_builders


class AdaMLPBlock(nn.Module):
    r"""Residual MLP block module with adaptive layer norm for conditioning.

    Arguments:
        hidden_dim: The dimensionality of the MLP block.
        cond_dim: The number of embedding features.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        mlp_ratio: int = 1,
        act = nn.GELU(),
    ):
        super().__init__()

        self.ada_ln = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        # Initialize the last layer to zero
        self.ada_ln[-1].weight.data *= 0.1
        self.ada_ln[-1].bias.data.zero_()

        # MLP block
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            act,
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Arguments:
            h: The input tensor, with shape (B, D_hidden).
            yt: The embedding vector, with shape (B, D_emb).

        Returns:
            The output tensor, with shape (B, D_x).
        """

        shift_, scale_, gate_ = self.ada_ln(cond).chunk(3, dim=-1)

        y = (scale_ + 1) * x + shift_
        y = self.block(y)
        y = x + gate_ * y

        return y


class GlobalEmbeddingMLP(nn.Module):
    """
    Global embedding MLP that outputs the conditioning embedding
    that is fed into the AdaMLPBlock.
    This MLP takes in the time embedding and a several channels of the condition
    and outputs a single (1D) embedding vector.

    Args:
        x_emb_dim: The dimensionality of the input tensor.
        cond_emb_dim: The dimensionality of the conditioning embedding.
        time_emb_dim: The dimensionality of the time embedding.
        hidden_dim: The dimensionality of the MLP block.
        num_intermediate_layers: Number of intermediate MLP blocks (Linear+GeLU+Linear).
        mlp_ratio: The ratio of the hidden dimension to the intermediate dimension.
        **kwargs: Key word arguments handed to the AdaMLPBlock.
    """

    def __init__(
        self,
        x_emb_channels: int,
        ctx_emb_channels: int,
        modes: int,
        time_emb_dim: int,
        cond_emb_dim: int,
        hidden_dim: int = 100,
        num_intermediate_layers: int = 0,
        mlp_ratio: int = 1,
        act = nn.GELU(),
        **kwargs,
    ):
        super().__init__()
        self.num_intermediate_layers = num_intermediate_layers

        self.x_pooling = nn.Conv1d(in_channels=x_emb_channels, out_channels=1, kernel_size=1)
        self.ctx_pooling = nn.Conv1d(in_channels=ctx_emb_channels, out_channels=1, kernel_size=1)


        self.mlp_blocks = nn.ModuleList()

        self.input_layer = nn.Linear(
            2*modes + time_emb_dim, hidden_dim
        )

        for _i in range(num_intermediate_layers):
            self.mlp_blocks.append(
                nn.Sequential(
                    act,
                    nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
                    act,
                    nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
                )
            )

        self.output_layer = nn.Linear(hidden_dim, cond_emb_dim)

    def forward(self, x_emb: Tensor, ctx_emb: Tensor ,t_emb: Tensor) -> Tensor:
        """
        Forward pass of the GlobalEmbeddingMLP.

        Args:
        x_emb: Condition tensor of shape (batch_size, x_emb_channels, modes).
        ctx_emb: Context tensor of shape (batch_Size, ctx_emb_channels, modes).
        t_emb: Time tensor of shape (batch_size, time_emb_dim).
        """

        x_pool = self.x_pooling(x_emb).squeeze(1) #(batch_size,modes)
        ctx_pool = self.ctx_pooling(ctx_emb).squeeze(1) #(batch_size,modes)
        cond_emb = torch.cat([x_pool, ctx_pool,t_emb], dim=-1)
        cond_emb = self.input_layer(cond_emb)
        for mlp_block in self.mlp_blocks:
            cond_emb = mlp_block(cond_emb)
        return self.output_layer(cond_emb)



# abstract class to ensure forward signature for flow matching networks
class VectorFieldNet(nn.Module, ABC):
    @abstractmethod
    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor: ...


class VectorFieldMLP(VectorFieldNet):
    """MLP for vector field estimation"""

    def __init__(
        self,
        input_dim: int,
        x_emb_channels: int,
        ctx_emb_channels: int,
        modes: int,
        time_emb_dim: int,
        cond_emb_dim: int,
        hidden_features: int = 64,
        num_layers: int = 1,
        global_mlp_ratio: int = 1,
        num_intermediate_mlp_layers: int = 0,
        adamlp_ratio: int = 1,
        act = nn.GELU(),
    ):
        """Initialize vector field MLP.

        Args:
            input_dim (int):
                Dimension of the input (theta or state).
            x_emb_channels (int):
                Number of channels in the conditioning x embedding.
            ctx_emb_channels (int):
                Number of channels in the context embedding.
            modes (int):
                Number of modes in the Fourier feature embedding.
            time_emb_dim (int):
                Dimension of the time embedding.
            condition_emb_dim (int):
                Dimension to embed all the conditioning info into.
            hidden_features (int):
                Number of hidden features in each layer. Defaults to 64.
            num_layers (int, optional):
                Number of layers in the network. Defaults to 1.
            global_mlp_ratio (int, optional):
                Ratio of the hidden dimension to the intermediate
                dimension in the global MLP. Defaults to 1.
            num_intermediate_mlp_layers (int, optional):
                Number of intermediate MLP blocks (Linear+GeLU+Linear)
                in the global MLP. Defaults to 0.
            adamlp_ratio (int, optional):
                Ratio of the hidden dimension to the intermediate
                dimension in the AdaMLPBlock. Defaults to 1.
            act (nn.Module, optional):
                Activation function. Defaults to nn.GELU.
            skip_connections (bool, optional):
                Whether to use skip connections. Defaults to True.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # Global MLP for time and condition embedding
        self.global_mlp = GlobalEmbeddingMLP(
            x_emb_channels = x_emb_channels,
            ctx_emb_channels = ctx_emb_channels,
            modes = modes,
            time_emb_dim = time_emb_dim,
            cond_emb_dim = cond_emb_dim,
            num_intermediate_layers=num_intermediate_mlp_layers,
            global_mlp_ratio=global_mlp_ratio,
            act = act,
        )
        self.input_dim = hidden_features

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_features))

        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(
                AdaMLPBlock(
                    hidden_dim=hidden_features,
                    cond_dim=cond_emb_dim,
                    mlp_ratio=adamlp_ratio,
                    act=act,
                )
            )

        # Output layer
        self.layers.append(nn.Linear(hidden_features, input_dim))

    def forward(self, theta: Tensor, x_emb: Tensor, ctx_emb: Tensor, t_emb: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            theta: Parameters (for FMPE) or state (for NPSE).
            x_emb: Continuous parameter condition.
            ctx_emb: Context condition.
            t_emb: Time embedding.

        Returns:
            Vector field evaluation at the provided points.
        """

        h = theta

        # Get condition embedding
        cond_emb = self.global_mlp(x_emb=x_emb, ctx_emb = ctx_emb, t_emb = t_emb)

        # Forward pass through MLP
        h = self.layers[0](h)  # input to hidden layer

        for layer in self.layers[1:-1]:  # hidden layers
            h = layer(h, cond_emb)

        h = self.layers[-1](h)  # hidden to output

        return h
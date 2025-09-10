import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fnope.nets.spectral_convolutions import SpectralConv1d_DSE,SpectralConv1d_DSE_FixedContext,SpectralConv2d_DSE,SpectralConv2d_DSE_FixedContext


class FNO1DBlock_DSE(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            out_channels: int,
            modes: int,
            act=nn.GELU(),
            time_embedding_channels: Optional[int] = None,
            pos_embedding_channels: Optional[int] = None,
            conditional_info_channels: Optional[int] = None,
    ):
        
        super().__init__()

        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.act = act
        self.time_embedding_channels = time_embedding_channels
        self.pos_embedding_channels = pos_embedding_channels
        self.conditional_info_channels = conditional_info_channels

        # Initialize spectral convolutional layer
        self.specConv_layer = SpectralConv1d_DSE(self.in_channels, self.out_channels, self.modes)

        # Initialize convolutional layer in physical domain
        self.w_layer = nn.Conv1d(self.in_channels, self.out_channels, 1)

        # Optional: Initialize layer for incorporating time embedding
        if time_embedding_channels is not None:
            self.temb_proj = nn.Linear(time_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating position embedding
        if pos_embedding_channels is not None:
            self.pos_proj = nn.Linear(pos_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating conditional information
        if conditional_info_channels is not None:
            self.conditional_info_proj = nn.Linear(conditional_info_channels, out_channels)

    def forward(self, x, time_embedding = None, pos_embedding = None, transform = None, conditional_info = None): # x should have shape (batch, n_points, channels)
        x_skip = x
        if time_embedding is not None: 
            v = self.temb_proj(self.act(time_embedding)).unsqueeze(1)
            x += v
        if pos_embedding is not None:
            p = self.pos_proj(self.act(pos_embedding))
            x += p
        if conditional_info is not None:
            c = self.conditional_info_proj(self.act(conditional_info)).unsqueeze(1)
            x += c
        x1 = self.specConv_layer(x, transform)
        
        x2 = self.w_layer(x_skip.permute(0, 2, 1))
        x = x1 + x2.permute(0, 2, 1)
        x = F.gelu(x)

        return x
    

class FNO1DBlock_DSE_FixedContext(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            ctx_channels: int,
            out_channels: int,
            modes: int,
            act=nn.GELU(),
            time_embedding_channels: Optional[int] = None,
            pos_embedding_channels: Optional[int] = None,
            conditional_info_channels: Optional[int] = None,
    ):
        
        super().__init__()

        self.modes = modes
        self.in_channels = in_channels
        self.ctx_channels = ctx_channels
        self.out_channels = out_channels 
        self.act = act
        self.time_embedding_channels = time_embedding_channels
        self.pos_embedding_channels = pos_embedding_channels
        self.conditional_info_channels = conditional_info_channels

        # Initialize spectral convolutional layer
        self.specConv_layer = SpectralConv1d_DSE_FixedContext(in_channels=self.in_channels,
                                                              ctx_channels=self.ctx_channels,
                                                              out_channels= self.out_channels,
                                                              modes=  self.modes)

        # Initialize convolutional layer in physical domain
        self.w_layer = nn.Conv1d(self.in_channels, self.out_channels, 1)

        # Optional: Initialize layer for incorporating time embedding
        if time_embedding_channels is not None:
            self.temb_proj = nn.Linear(time_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating position embedding
        if pos_embedding_channels is not None:
            self.pos_proj = nn.Linear(pos_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating conditional information
        if conditional_info_channels is not None:
            self.conditional_info_proj = nn.Linear(conditional_info_channels, out_channels)

        

    def forward(self, x, ctx, time_embedding = None, pos_embedding = None, transform = None, ctx_transform=None , conditional_info = None): # x should have shape (batch, n_points, channels)
        x_skip = x
        if time_embedding is not None: 
            v = self.temb_proj(self.act(time_embedding)).unsqueeze(1)
            x += v
        if pos_embedding is not None:
            p = self.pos_proj(self.act(pos_embedding))
            x += p
        if conditional_info is not None:
            c = self.conditional_info_proj(self.act(conditional_info)).unsqueeze(1)
            x += c
        x1 = self.specConv_layer(x, ctx, transform, ctx_transform)
        
        x2 = self.w_layer(x_skip.permute(0, 2, 1))
        x = x1 + x2.permute(0, 2, 1)
        x = F.gelu(x)

        return x

class FNO2DBlock_DSE(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            out_channels: int,
            modes: int,
            equispaced:bool = False,
            act=nn.GELU(),
            time_embedding_channels: Optional[int] = None,
            pos_embedding_channels: Optional[int] = None,
            conditional_info_channels: Optional[int] = None,
    ):
        
        super().__init__()

        self.modes = modes
        self.equispaced = equispaced
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.act = act
        self.time_embedding_channels = time_embedding_channels
        self.pos_embedding_channels = pos_embedding_channels
        self.conditional_info_channels = conditional_info_channels

        # Initialize spectral convolutional layer
        self.specConv_layer = SpectralConv2d_DSE(in_channels=self.in_channels,
                                                 out_channels= self.out_channels,
                                                 modes = self.modes,
                                                 equispaced = self.equispaced)

        # Initialize convolutional layer in physical domain
        if self.equispaced:
            self.w_layer = nn.Conv2d(self.in_channels, self.out_channels, 1)
        else:
            self.w_layer = nn.Conv1d(self.in_channels, self.out_channels, 1)

        # Optional: Initialize layer for incorporating time embedding
        if time_embedding_channels is not None:
            self.temb_proj = nn.Linear(time_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating position embedding
        if pos_embedding_channels is not None:
            self.pos_proj = nn.Linear(pos_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating conditional information
        if conditional_info_channels is not None:
            self.conditional_info_proj = nn.Linear(conditional_info_channels, out_channels)

        self.view_shape = (-1, self.out_channels, 1 , 1) if self.equispaced else (-1, 1, self.out_channels)
        self.permute_shape = (0, 1, 2, 3) if self.equispaced else (0, 2, 1)

    def forward(self, x, time_embedding = None, pos_embedding = None, transform = None, conditional_info = None): # x should have shape (batch, n_points, channels)

        x_skip = x
        if time_embedding is not None: 
            v = self.temb_proj(self.act(time_embedding)).view(*self.view_shape)
            x += v
        if pos_embedding is not None:
            p = self.pos_proj(self.act(pos_embedding))
            x += p
        if conditional_info is not None:
            c = self.conditional_info_proj(self.act(conditional_info)).view(*self.view_shape)
            x += c
        x1 = self.specConv_layer(x, transform)
        x2 = self.w_layer(x_skip.permute(*self.permute_shape))
        x = x1 + x2.permute(*self.permute_shape)
        x = F.gelu(x)

        return x
    

class FNO2DBlock_DSE_FixedContext(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            ctx_channels: int,
            out_channels: int,
            modes: int,
            equispaced:bool = False,
            act=nn.GELU(),
            time_embedding_channels: Optional[int] = None,
            pos_embedding_channels: Optional[int] = None,
            conditional_info_channels: Optional[int] = None,
    ):
        
        super().__init__()

        self.modes = modes
        self.equispaced = equispaced
        self.in_channels = in_channels
        self.ctx_channels = ctx_channels
        self.out_channels = out_channels 
        self.act = act
        self.time_embedding_channels = time_embedding_channels
        self.pos_embedding_channels = pos_embedding_channels
        self.conditional_info_channels = conditional_info_channels

        # Initialize spectral convolutional layer
        self.specConv_layer = SpectralConv2d_DSE_FixedContext(in_channels=self.in_channels,
                                                              ctx_channels=self.ctx_channels,
                                                              out_channels= self.out_channels,
                                                              modes=  self.modes,
                                                              equispaced=self.equispaced)

        # Initialize convolutional layer in physical domain
        if self.equispaced:
            self.w_layer = nn.Conv2d(self.in_channels, self.out_channels, 1)
        else:
            self.w_layer = nn.Conv1d(self.in_channels, self.out_channels, 1)
            
        # Optional: Initialize layer for incorporating time embedding
        if time_embedding_channels is not None:
            self.temb_proj = nn.Linear(time_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating position embedding
        if pos_embedding_channels is not None:
            self.pos_proj = nn.Linear(pos_embedding_channels, out_channels)
        # Optional: Initialize layer for incorporating conditional information
        if conditional_info_channels is not None:
            self.conditional_info_proj = nn.Linear(conditional_info_channels, out_channels)

        
        self.view_shape = (-1, self.out_channels, 1 , 1) if self.equispaced else (-1, 1, self.out_channels)
        self.permute_shape = (0, 1, 2, 3) if self.equispaced else (0, 2, 1)

    def forward(self, x, ctx, time_embedding = None, pos_embedding = None, transform = None, ctx_transform=None , conditional_info = None): # x should have shape (batch, n_points, channels)
        x_skip = x
        if time_embedding is not None: 
            v = self.temb_proj(self.act(time_embedding)).view(*self.view_shape)
            x += v
        if pos_embedding is not None:
            p = self.pos_proj(self.act(pos_embedding))
            x += p
        if conditional_info is not None:
            c = self.conditional_info_proj(self.act(conditional_info)).view(*self.view_shape)
            x += c
        x1 = self.specConv_layer(x, ctx, transform, ctx_transform)
        x2 = self.w_layer(x_skip.permute(*self.permute_shape))
        x = x1 + x2.permute(*self.permute_shape)
        x = F.gelu(x)

        return x
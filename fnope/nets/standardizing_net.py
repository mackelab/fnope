import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class FilterStandardizing(nn.Module):
    """
    Standardizes the input tensor using a filter-based normalization.
    Calculate the power and mean per channel, use it as a shift_and_scale transformation.
    It is calculated directly from the training data. It then applies it per batch.
    """
    def __init__(self,
                 x:Tensor,
                 point_positions:Optional[Tensor],
                 num_channels:int,
                 modes:int, 
                 cutoff=False, 
                 eps=1e-6,
                 **kwargs
                 ):
        
        super().__init__()
        self.num_channels = num_channels
        self.cutoff = cutoff
        self.eps = eps
        self.n_modes = modes


        channelwise_mean,channelwise_power = self.calculate_channelwise_power(x,point_positions)
        self.register_buffer('channelwise_mean', channelwise_mean)
        self.register_buffer('channelwise_power', channelwise_power)

    def generalized_filter_matrix(self,batchsize:int,point_positions: Optional[Tensor], height:int) -> torch.Tensor:
        """
        Constructs a low-pass filter matrix for non-equally spaced grid points.
        """

        if point_positions is not None:
            assert point_positions.shape[-1] == height, f"point_positions should have shape (batchsize, height), got {point_positions.shape}"
        elif point_positions is None:
            point_positions = torch.linspace(0, 1, height, device=self.device)
        assert point_positions.ndim in [1,2], f"point_positions should be 1D or 2D tensor, got {point_positions.ndim}D tensor"
        if point_positions.ndim == 1:
            point_positions = point_positions.unsqueeze(0).expand(batchsize, -1)
        elif point_positions.ndim == 2:
            if point_positions.shape[0] == 1:
                point_positions = point_positions.expand(batchsize, -1)

        deltas = torch.diff(point_positions, prepend=point_positions[:,:1],)
        local_bandwidths = (torch.roll(deltas, shifts=-1,dims=(-1,)) + deltas) / 2  # Local avg spacing

        # Define per-point bandwidths
        bandwidths = self.n_modes * local_bandwidths / height

        # Construct adaptive Gaussian weight matrix
        H = torch.exp(-((point_positions.unsqueeze(2) - point_positions.unsqueeze(1)) ** 2) / (2 * bandwidths[:,:,None] ** 2))

        # Normalize each row to preserve energy
        H /= H.sum(dim=1, keepdim=True)

        return H

    def standardize_1d(self,
                    x:torch.Tensor,
                    point_positions:Optional[Tensor],
                    ):
        """
        Standardizes the input tensor per channel.
        If `cutoff` is True, applies a low-pass filter to the input tensor first.
        
        Args:
        x (torch.Tensor): Input tensor of shape (batchsize, channels, height).
        point_positions (torch.Tensor): Tensor of shape (batchsize, height) or (height) representing the positions of the points.
        """
        batchsize, channels, height = x.shape
        channelwise_mean = self.channelwise_mean.view(1, self.num_channels, 1)
        channelwise_power = self.channelwise_power.view(1, self.num_channels, 1)
        x = x - channelwise_mean
        if self.cutoff:
            self._H = self.generalized_filter_matrix(batchsize, point_positions, height)

            x = torch.einsum('ijk,ikl->ijl', x, self._H)
        out = x/ channelwise_power

        return out
    
    def unstandardize_1d(self,
                        x:torch.Tensor,
                        point_positions:Optional[Tensor]):
        """
        Unstandardizes the input tensor per channel.

        Args:
        x (torch.Tensor): Input tensor of shape (batchsize, channels, height).
        point_positions (torch.Tensor): Tensor of shape (batchsize, height) or (height) representing the positions of the points.
                                        This is ignored, but included in the functional call for consistency.
        """
        batchsize, channels, height = x.shape
        channelwise_mean = self.channelwise_mean.view(1, self.num_channels, 1)
        channelwise_power = self.channelwise_power.view(1, self.num_channels, 1)
        out = x * channelwise_power
        out = out + channelwise_mean

        return out
    

    def calculate_channelwise_power(self,
                                    x: torch.Tensor,
                                    point_positions:Optional[Tensor]):
        """
        Calculates the channelwise mean and power of the input tensor.
        Args:
        x (torch.Tensor): Input tensor of shape (batchsize, channels, height).
        point_positions (torch.Tensor): Tensor of shape (batchsize, height) or (height) representing the positions of the points.
        """

        channelwise_mean = torch.mean(x, dim=(0, 2), keepdim=True)
        variance = torch.var(x, dim=(2,), keepdim=True)
        avg_variance = torch.mean(variance, dim=(0,), keepdim=True)
        avg_sd = torch.sqrt(avg_variance+self.eps)
        return channelwise_mean,avg_sd
     
    def standardize(self, x: torch.Tensor, point_positions: Optional[Tensor]):
        return self.standardize_1d(x, point_positions)
    
    def unstandardize(self, x: torch.Tensor, point_positions: Optional[Tensor]):
        return self.unstandardize_1d(x, point_positions)

    def forward(self, x: torch.Tensor,point_positions:Optional[Tensor]):
        dim = x.dim()
        if dim == 3:
            return self.standardize(x,point_positions)
        else:
            raise NotImplementedError
        


        
class FiniteStandardizing(nn.Module):
    """
    Normalizing finite (non-continuous) data by subtracting the mean and dividing by the standard deviation.
    The interface matches the interface of the continuous standardizing classes.
    
    """
    def __init__(self,x, eps=1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        mean,sd = self.calculate_mean_sd(x)
        self.register_buffer('mean', mean)
        self.register_buffer('sd', sd)


    def standardize(self, x:Tensor):
        x = x - self.mean
        out = x/ self.sd
        return out
    
    def unstandardize(self, x:Tensor):
        out = x * self.sd
        out = out + self.mean
        return out
    

    def calculate_mean_sd(self, x:Tensor):
        mean = torch.mean(x,dim=0)
        sd = torch.clip(torch.std(x,dim=0),min=self.eps)
        return mean,sd
    
    def forward(self, x: torch.Tensor):
        return self.standardize_1d(x)
    

class FilterStandardizing2d(nn.Module):
    """
    Standardizes the input tensor using a filter-based normalization.
    Calculate the power and mean per channel, use it as a shift_and_scale transformation.
    It is calculated directly from the training data. It then applies it per batch.
    """
    def __init__(self,
                 x:Tensor,
                 point_positions:Optional[Tensor],
                 num_channels:int,
                 modes:int, 
                 cutoff=False, 
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.modes = modes
        self.fft_norm = 'forward'
        self.cutoff = cutoff
        self.eps = eps

        channelwise_mean,channelwise_power = self.calculate_channelwise_power(x,point_positions)
        self.register_buffer('channelwise_mean', channelwise_mean)
        self.register_buffer('channelwise_power', channelwise_power)



    def standardize_2d(self, x:torch.Tensor, point_positions: Optional[torch.Tensor]):
        if x.ndim == 3:
            channelwise_mean = self.channelwise_mean.view(1, self.num_channels, 1)
            channelwise_power = self.channelwise_power.view(1, self.num_channels, 1)

        elif x.ndim == 4:
            channelwise_mean = self.channelwise_mean.view(1, self.num_channels, 1, 1)
            channelwise_power = self.channelwise_power.view(1, self.num_channels, 1, 1)
        x = x - channelwise_mean
        
        out = x / channelwise_power

        return out

    def unstandardize_2d(self, x: torch.Tensor, point_positions: Optional[torch.Tensor]):
        
        if x.ndim == 3:
            channelwise_mean = self.channelwise_mean.view(1, self.num_channels, 1)
            channelwise_power = self.channelwise_power.view(1, self.num_channels, 1)

        elif x.ndim == 4:
            channelwise_mean = self.channelwise_mean.view(1, self.num_channels, 1, 1)
            channelwise_power = self.channelwise_power.view(1, self.num_channels, 1, 1)
        out = x * channelwise_power
        out = out + channelwise_mean

        return out
    
    def standardize(self, x: torch.Tensor, point_positions: Optional[torch.Tensor]):
        return self.standardize_2d(x, point_positions)
    def unstandardize(self, x: torch.Tensor, point_positions: Optional[torch.Tensor]):
        return self.unstandardize_2d(x, point_positions)
    
    def calculate_channelwise_power(self, x: torch.Tensor, point_positions: Optional[torch.Tensor]):

        # THIS ASSUMES THAT ALL SAMPLES IN BATCH ARE ON SAME GRID
        if x.ndim == 3:
            channelwise_mean = torch.mean(x, dim=(0, 2), keepdim=True)
            variance = torch.var(x, dim=(2,), keepdim=True)
            avg_variance = torch.mean(variance, dim=(0,), keepdim=True)
            avg_sd = torch.sqrt(avg_variance+self.eps)
        elif x.ndim == 4:
            channelwise_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            variance = torch.var(x, dim=(2,3), keepdim=True)
            avg_variance = torch.mean(variance, dim=(0,), keepdim=True)
            avg_sd = torch.sqrt(avg_variance+self.eps)

        else:
            raise ValueError("Input tensor must be 3D or 4D.")
        
        return channelwise_mean,avg_sd
    
    def forward(self, x: torch.Tensor, point_positions:Optional[torch.Tensor]):
        dim = x.dim()
        if dim == 3 or dim == 4:
            return self.standardize_2d(x, point_positions)
        else:
            raise NotImplementedError



class IdentityStandardizing(nn.Module):
    """
    Identity standardizing class that does not perform any normalization.
    This is useful for debugging or when no normalization is needed.
    """
    def __init__(self):
        super().__init__()

    def standardize(self, x, **kwargs):
        return x
    
    def standardize_1d(self, x, **kwargs):
        return x
    
    def standardize_2d(self, x,  **kwargs):
        return x

    def forward(self, x,  **kwargs):
        return x
    
    def unstandardize(self, x,  **kwargs):
        return x
    def unstandardize_1d(self, x,  **kwargs):
        return x
    def unstandardize_2d(self, x,  **kwargs):
        return x

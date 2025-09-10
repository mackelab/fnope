import torch
import torch.nn as nn
from fnope.nets.spectral_transforms import FFT1D, VFT1D, FFT2D, VFT2D
from torch import Tensor
from typing import Union

class SpectralConv1d_DSE(nn.Module):
    """
    A 1D spectral convolutional layer using the Fourier transform.
    This layer applies a learned complex multiplication in the frequency domain.

    Adapted from:
    - Lingsch et al. (2024) FUSE: Fast Unified Simulation and Estimation for PDEs
    - Li et al. (2021) Fourier Neural Operator for Parametric Partial Differential
                        Equations

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to multiply,
            at most floor(N/2) + 1.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # for equally spaced data, all Fourier modes must be passed 
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, self.modes, out_channels, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: Tensor, weights: Tensor) -> Tensor:
        """
        Performs complex multiplication in the Fourier domain.

        Args:
            input: Input tensor of shape (batch, modes, in_channels).
            weights: Weight tensor of shape (in_channels,  modes, out_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch,  modes, out_channels).
        """

        return torch.einsum("bxi,ixo->bxo", input, weights)

    def forward(self, x: Tensor, transform: Union[FFT1D,VFT1D], norm="ortho") -> Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x: Input tensor of shape (batch, n_points, in_channels).
            transform: Fourier-like transform operator with forward and inverse methods.
            norm: Normalization type for the Fourier transform. Default is 'ortho'.

        Returns:
            The real part of the transformed output tensor
            with shape (batch, points, out_channels).
        """
        # Compute Fourier coefficients

        x_ft = transform.forward(x, norm=norm)
        out_ft = self.compl_mul1d(x_ft, self.weights)
        # Return to physical space
        x = transform.inverse(out_ft, norm=norm)


        return x.real # output from irfft should be real anyway, this is only required for non-equispaced approach


class SpectralConv1d_DSE_FixedContext(nn.Module):
    """
    A 1D spectral convolutional layer using the Fourier transform.
    This layer applies a learned complex multiplication in the frequency domain.

    Adapted from:
    - Lingsch et al. (2024) FUSE: Fast Unified Simulation and Estimation for PDEs
    - Li et al. (2021) Fourier Neural Operator for Parametric Partial Differential
                        Equations

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to multiply,
            at most floor(N/2) + 1.
    """

    def __init__(self, in_channels: int, ctx_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.in_channels = in_channels
        self.ctx_channels = ctx_channels
        self.out_channels = out_channels
        self.modes = modes # for equally spaced data, all Fourier modes must be passed 
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels + ctx_channels, self.modes, out_channels, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: Tensor, ctx: Tensor, weights: Tensor) -> Tensor:
        """
        Performs complex multiplication in the Fourier domain.

        Args:
            input: Input tensor of shape (batch, modes, in_channels).
            ctx: Context tensor of shape (batch, modes, ctx_channels).
            weights: Weight tensor of shape (in_channels, modes, out_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch, modes, out_channels).
        """
        # Concatenate input and context along the channel dimension
        input = torch.cat((input, ctx), dim=2)
        return torch.einsum("bmi,imo->bmo", input, weights)

    def forward(self, x: Tensor, ctx: Tensor, transform: Union[FFT1D,VFT1D], ctx_transform: Union[FFT1D,VFT1D],norm="ortho") -> Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x: Input tensor of shape (batch, n_points, in_channels).
            ctx: Context tensor of shape (batch, nctx_points, ctx_channels).
            transform: Fourier-like transform operator with forward and inverse methods.
            ctx_transform: Fourier-like transform operator for context with forward and inverse methods.
            norm: Normalization type for the Fourier transform. Default is 'ortho'.

        Returns:
            The real part of the transformed output tensor
            with shape (batch, points, out_channels).
        """

        x_ft = transform.forward(x, norm=norm)
        ctx_ft = ctx_transform.forward(ctx, norm=norm)
        out_ft = self.compl_mul1d(x_ft, ctx_ft, self.weights)

        # Return to physical space
        x = transform.inverse(out_ft, norm=norm)

 

        return x.real # output from irfft should be real anyway, this is only required for non-equispaced approach



class SpectralConv2d_DSE(nn.Module):
    """
    A spectral convolutional layer using the Fourier transform.
    This layer applies a learned complex multiplication in the frequency domain.

    Adapted from:
    - Lingsch et al. (2024) FUSE: Fast Unified Simulation and Estimation for PDEs
    - Li et al. (2021) Fourier Neural Operator for Parametric Partial Differential
                        Equations

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to multiply,
            at most floor(N/2) + 1.
        equispaced: Boolean indicating whether the data is equispaced or not.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int, equispaced: bool):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # for equally spaced data, all Fourier modes must be passed 
        self.scale = (1 / (in_channels * out_channels))
        self.equispaced = equispaced

        # Initialize weights 
        if self.equispaced: 
            self.weights = nn.Parameter(
                self.scale
                * torch.rand(in_channels, out_channels,  self.modes, self.modes,  dtype=torch.cfloat)
            )
        else:  
            self.weights = nn.Parameter(
                self.scale
                * torch.rand(in_channels, self.modes**2, out_channels, dtype=torch.cfloat)
            )

    # Complex multiplication 1D
    def compl_mul1d(self, input: Tensor, weights: Tensor) -> Tensor:
        """
        Performs complex multiplication in the Fourier domain.

        Args:
            input: Input tensor of shape (batch, modes, in_channels).
            weights: Weight tensor of shape (in_channels,  modes, out_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch,  modes, out_channels).
        """

        return torch.einsum("bxi,ixo->bxo", input, weights)

    # Complex multiplication 2D
    def compl_mul2d(self, input, weights):
        
        # (batch,in_channel,modes_x,modes_y), (in_channel, out_channels, modes_x, modes_y,) -> (batch,out_channels, modes,modes, )
        
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def compl_mul(self,input, weights):
        if self.equispaced:
            return self.compl_mul2d(input, weights)
        else:
            return self.compl_mul1d(input, weights)

    def forward(self, x: Tensor, transform: Union[FFT2D,VFT2D],norm="ortho") -> Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x: Input tensor of shape (batch, number_points, in_channels) for non-equispaced data 
            (transform = VFT2D) or (batch, nx_points, ny_points, in_channels) for equispaced data
            (transform = FFT2D).
            transform: Fourier-like transform operator with forward and inverse methods.
            norm: Normalization type for the Fourier transform. Default is 'ortho'.

        Returns:
            The real part of the transformed output tensor
            with shape (batch, points, out_channels) if non-equispaced
            or shape (batch,nx_points,ny_points,out_channels) if equispaced.
        """

        if isinstance(transform, VFT2D):
            assert x.ndim == 3, f"Given non-equispaced data, input tensor x must have shape (batch, number_points, in_channels) but got {x.shape}."
        elif isinstance(transform, FFT2D):
            assert x.ndim == 4, f"Given equispaced data, input tensor x must have shape (batch, nx_points, ny_points, in_channels) but got {x.shape}."
        
        
        # Compute Fourier coefficients
        x_ft = transform.forward(x, norm=norm)
        out_ft = self.compl_mul(x_ft, self.weights)
        # Return to physical space
        x = transform.inverse(out_ft, norm=norm)


        return x.real # output from irfft should be real anyway, this is only required for non-equispaced approach
    

class SpectralConv2d_DSE_FixedContext(nn.Module):
    """
    A spectral convolutional layer using the Fourier transform.
    This layer applies a learned complex multiplication in the frequency domain.

    Adapted from:
    - Lingsch et al. (2024) FUSE: Fast Unified Simulation and Estimation for PDEs
    - Li et al. (2021) Fourier Neural Operator for Parametric Partial Differential
                        Equations

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to multiply,
            at most floor(N/2) + 1.
        equispaced: Boolean indicating whether the data is equispaced or not.
    """

    def __init__(self, in_channels: int, ctx_channels: int, out_channels: int, modes: int, equispaced: bool):
        super().__init__()

        self.in_channels = in_channels
        self.ctx_channels = ctx_channels
        self.out_channels = out_channels
        self.modes = modes # for equally spaced data, all Fourier modes must be passed 
        self.scale = (1 / (in_channels * out_channels))
        self.equispaced = equispaced

        # Initialize weights 
        if self.equispaced: 
            self.weights = nn.Parameter(
                self.scale
                * torch.rand(in_channels + ctx_channels, out_channels,  self.modes, self.modes,  dtype=torch.cfloat)
            )
        else:  
            self.weights = nn.Parameter(
                self.scale
                * torch.rand(in_channels + ctx_channels, self.modes**2, out_channels, dtype=torch.cfloat)
            )

    # Complex multiplication 1D
    def compl_mul1d(self, input: Tensor, ctx: Tensor, weights: Tensor) -> Tensor:
        """
        Performs complex multiplication in the Fourier domain.

        Args:
            input: Input tensor of shape (batch, modes, in_channels).
            ctx: Context tensor of shape (batch, modes, ctx_channels).
            weights: Weight tensor of shape (in_channels + ctx_channels,  modes, out_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch,  modes, out_channels).
        """
        # Concatenate input and context along the channel dimension
        input = torch.cat((input, ctx), dim=2)
        return torch.einsum("bxi,ixo->bxo", input, weights)
    
    # Complex multiplication 2D
    def compl_mul2d(self, input: Tensor, ctx: Tensor, weights: Tensor) -> Tensor:
        """
        Performs complex multiplication in the Fourier domain.

        Args:
            input: Input tensor of shape (batch, in_channels, modes, modes).
            ctx: Context tensor of shape (batch, ctx_channels, modes, modes).
            weights: Weight tensor of shape (in_channels + ctx_channels, out_channels,  modes, modes).

        Returns:
            torch.Tensor: Output tensor of shape (batch,out_channels, modes, modes).
        """
        # Concatenate input and context along the channel dimension
        input = torch.cat((input, ctx), dim=1)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def compl_mul(self, input: Tensor, ctx: Tensor, weights: Tensor) -> Tensor:
        if self.equispaced:
            return self.compl_mul2d(input, ctx, weights)
        else:
            return self.compl_mul1d(input, ctx, weights)
        
    def forward(self, x: Tensor, ctx: Tensor, transform: Union[FFT2D,VFT2D], ctx_transform: Union[FFT2D,VFT2D],norm="ortho") -> Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x: Input tensor of shape (batch, number_points, in_channels) for non-equispaced data 
            (transform = VFT2D) or (batch, nx_points, ny_points, in_channels) for equispaced data
            (transform = FFT2D).
            ctx: Context tensor of shape (batch, nctx_points, ctx_channels).
            transform: Fourier-like transform operator with forward and inverse methods.
            ctx_transform: Fourier-like transform operator for context with forward and inverse methods.
            norm: Normalization type for the Fourier transform. Default is 'ortho'.

        Returns:
            The real part of the transformed output tensor
            with shape (batch, points, out_channels) if non-equispaced
            or shape (batch,nx_points,ny_points,out_channels) if equispaced.
        """

        if isinstance(transform, VFT2D):
            assert x.ndim == 3, f"Given non-equispaced data, input tensor x must have shape (batch, number_points, in_channels) but got {x.shape}."
        elif isinstance(transform, FFT2D):
            assert x.ndim == 4, f"Given equispaced data, input tensor x must have shape (batch, nx_points, ny_points, in_channels) but got {x.shape}."
        
        
        # Compute Fourier coefficients
        x_ft = transform.forward(x, norm=norm)
        ctx_ft = ctx_transform.forward(ctx, norm=norm)
        out_ft = self.compl_mul(x_ft, ctx_ft, self.weights)
        # Return to physical space
        x = transform.inverse(out_ft, norm=norm)


        return x.real # output from irfft should be real anyway, this is only required for non-equispaced approach
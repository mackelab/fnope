import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod

class SpectralTransform(ABC):
    def __init__(self, modes: int, device: Optional[Union[str, torch.device]] = "cpu"):
        self.modes = modes
        self.device= device

    @abstractmethod
    def forward(self, x: Tensor, norm: str = 'ortho') -> Tensor:
        pass

    @abstractmethod
    def inverse(self, x: Tensor, norm: str = 'ortho') -> Tensor:
        pass

class FFT1D(SpectralTransform):
    def __init__(
        self,
        modes: int,
        device: Optional[Union[str, torch.device]] = "cpu"
    ):
        super().__init__(modes, device)
        
        self.modes = modes
        self.device= device


    def forward(self, x: Tensor, norm: str = 'ortho') -> Tensor:
        """Perform forward Fourier transformation
        Args:
            data: Input data with shape (batch_size, n_points, conv_channel)
        """

        x_ft = torch.fft.rfft(x, dim=1, norm=norm)
        n_points = x.shape[1]
        self.n_points = n_points #save number of points for inverse transform

        n_modes = np.floor(n_points).astype(int) + 1
        out_ft = torch.zeros(x.shape[0], n_modes, x.shape[2], device=x_ft.device, dtype = torch.cfloat) 
        out_ft = x_ft[:, :self.modes, :]
        return out_ft
    

    def inverse(self, x: Tensor, norm: str = 'ortho') -> Tensor:
        """Perform inverse Fourier transformation
        Args:
            data: Input data with shape (batch_size, n_modes, conv_channel)
        """
        
        n_points = self.n_points

        x = torch.fft.irfft(x, n=n_points, dim=1, norm=norm)

        return x


class VFT1D(SpectralTransform):
    """Class for performing Fourier transformations for non-equally
    and equally spaced 1d grids.

    It provides a function for creating grid-dependent operator V to compute the
    Forward Fourier transform X of data x with X = V*x.
    The inverse Fourier transform can then be computed by x = V_inv*X with
    V_inv = transpose(conjugate(V)).

    Adapted from: Lingsch et al. (2024) Beyond Regular Grids: Fourier-Based
    Neural Operators on Arbitrary Domains

    Args:
        batch_size: Training batch size
        n_points: Number of 1d grid points
        modes: number of Fourier modes that should be used
            (maximal floor(n_points/2) + 1)
        point_positions: Grid point positions of shape (batch_size, n_points).
            If not provided, equispaced points are used. Positions have to be
            normalized with domain length.
    """

    def __init__(
        self,
        batch_size: int,
        n_points: int,
        modes: int,
        point_positions: Optional[Tensor] = None,
        device: Optional[Union[str, torch.device]] = "cpu"
    ):
        super().__init__(modes, device)
        self.number_points = n_points
        self.batch_size = batch_size
        self.modes = modes
        self.device= point_positions.device if point_positions is not None else device
        with torch.no_grad():
            if point_positions is not None:
                new_times = point_positions[:, None, :].to(self.device)
            else:
                new_times = (
                    (torch.arange(self.number_points) / self.number_points).repeat(
                        self.batch_size, 1
                    )
                )[:, None, :].to(device)

            self.new_times = new_times * 2 * np.pi

            self.X_ = torch.arange(modes).repeat(self.batch_size, 1)[:, :, None].float().to(device)
            # V_fwd: (batch, modes, points) V_inf: (batch, points, modes)
            self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self) -> Tuple[Tensor, Tensor]:
        """Create matrix operators V and V_inf for forward and backward
        Fourier transformation on arbitrary grids
        """

        X_mat = torch.bmm(self.X_, self.new_times)
        forward_mat = torch.exp(-1j * (X_mat))

        inverse_mat = torch.conj(forward_mat.clone()).permute(0, 2, 1)

        return forward_mat, inverse_mat

    def forward(self, data: Tensor, norm: str = 'ortho') -> Tensor:
        """Perform forward Fourier transformation
        Args:
            data: Input data with shape (batch_size, n_points, conv_channel)
        """

        data_fwd = torch.bmm(self.V_fwd, data.to(torch.complex64))  # (batch, modes, conv_channels)
        if norm == 'forward':
            data_fwd /= self.number_points
        elif norm == 'ortho':
            data_fwd /= np.sqrt(self.number_points)
        elif norm == 'backward':
            data_fwd /= 1.0

        return data_fwd

    def inverse(self, data: Tensor, norm: str = 'ortho') -> Tensor:
        """Perform inverse Fourier transformation
        Args:
            data: Input data with shape (batch_size, modes, conv_channel)
        """
        data_inv = torch.bmm(self.V_inv, data)  # (batch, n_points, conv_channels)
        if norm == 'backward':
            data_inv /= self.number_points
        elif norm == 'ortho':
            data_inv /= np.sqrt(self.number_points)
        elif norm == 'forward':
            data_inv /= 1.0

        return data_inv
    

class FFT2D(SpectralTransform):
    def __init__(
        self,
        modes: int,
        device: Optional[Union[str, torch.device]] = "cpu"
    ):
        super().__init__(modes, device)
        
        self.modes = modes
        self.device= device


    def forward(self, x: Tensor, norm: str = 'ortho') -> Tensor:
        """Perform forward Fourier transformation
        Args:
            data: Input data with shape (batch_size, nx_points, ny_points, conv_channel)
        """

        x_ft = torch.fft.rfft2(x, dim=(2, 3), norm=norm)
        self.nx_points,self.ny_points = x.shape[2],x.shape[3]


        nx_modes,ny_modes = self.nx_points, np.floor(self.ny_points).astype(int) + 1
        out_ft = torch.zeros(x.shape[0], x.shape[1], nx_modes,ny_modes, device=x.device, dtype = torch.cfloat) 
        out_ft = x_ft[:, :, :self.modes, :self.modes]
        return out_ft
    

    def inverse(self, x: Tensor, norm: str = 'ortho') -> Tensor:
        """Perform inverse Fourier transformation
        Args:
            data: Input data with shape (batch_size, n_modes, n_modes,conv_channel)
        """
        
        nx_points,ny_points = self.nx_points,self.ny_points

        x = torch.fft.irfft2(x, s=(nx_points,ny_points), dim=(2,3), norm=norm)
        return x


class VFT2D(SpectralTransform):
    """Class for performing Fourier transformations for non-equally
    spaced 2D grids.

    Adapted from: Lingsch et al. (2024) Beyond Regular Grids: Fourier-Based
    Neural Operators on Arbitrary Domains

    Args:
        x_positions: contains the x-positions of the samples (batch, n_points, channels(?))
        y_positions: contains the y-positions of the samples (batch, n_points, channels(?))

        2D data would then be passed in a flattened way

        The positions must be normalized with domain length, hence they have to be in 
        interval [0;1]
    """

    def __init__(self,
                 batch_size: int,
                 n_points: int,
                 modes: int,
                 point_positions: Tensor,
                 device: Optional[Union[str, torch.device]] = "cpu"):
        super().__init__(modes, device)
        self.number_points = n_points
        self.batch_size = batch_size
        self.modes = modes
        self.device= point_positions.device if point_positions is not None else device
        with torch.no_grad():
            x_positions = point_positions[:, :, 0].to(self.device)
            y_positions = point_positions[:, :, 1].to(self.device)

            # Positions need to b in interval [0; 2*pi]
            self.x_positions = x_positions * 2 * np.pi # CHECK THAT MULTIPLICATION WITH NP.PI WORKS PROPERLY 
            self.y_positions = y_positions * 2 * np.pi # (batch, number_points)
        

            modes_x = torch.arange(modes).repeat(modes)
            modes_y = torch.arange(modes).repeat_interleave(modes)
            self.X_ = modes_x.expand(self.batch_size, -1)[:,:,None].float().to(self.device) # (batch, modes**2, 1)
            self.Y_ = modes_y.expand(self.batch_size, -1)[:,:,None].float().to(self.device) # (batch, modes**2, 1)
            self.V_fwd, self.V_inv = self.make_matrix()



    def make_matrix(self) -> Tuple[Tensor, Tensor]:
            
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]) # (batch, modes**2, number_points)
        Y_mat = torch.bmm(self.Y_, self.y_positions[:,None,:]) # (batch, modes**2, number_points)
            
        forward_mat = torch.exp(-1j* (X_mat + Y_mat)).to(self.device) # (batch, modes**2, number_points)
        inverse_mat = torch.conj(forward_mat).permute(0,2,1).to(self.device) # (batch, modes**2, number_points)

        return forward_mat, inverse_mat
    

    def forward(self, data: Tensor, norm:str = 'ortho') -> Tensor:
        """Perform forward Fourier transformation
        Args:
            data: Input data with shape (batch_size, number_points, conv_channel)
        """

        data_fwd = torch.bmm(self.V_fwd, data.to(torch.complex64)) # (batch, modes**2, conv_channels)
        if norm == 'forward':
            data_fwd /= (self.number_points/2)
        elif norm == 'ortho':
            data_fwd /= np.sqrt(self.number_points/2)
        elif norm == 'backward':
            data_fwd /= 1.0

        return data_fwd

    def inverse(self, data: Tensor, norm: str = 'ortho') -> Tensor:
        """Perform inverse Fourier transformation
        Args:
            data: Input data with shape (batch_size, modes**2, conv_channel)
        """
        data_inv = torch.bmm(self.V_inv, data)  # (batch, number_points, conv_channels)
        if norm == 'backward':
            data_inv /= (self.number_points/2)
        elif norm == 'ortho':
            data_inv /= np.sqrt(self.number_points/2)
        elif norm == 'forward':
            data_inv /= 1.0

        return data_inv

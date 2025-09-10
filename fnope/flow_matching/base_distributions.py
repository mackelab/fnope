import numpy as np
import math
import torch
import torch.distributions as distributions
from torch.distributions import MultivariateNormal as MVN, Independent
from torch.distributions import Normal
from torch import Tensor
from typing import Optional

class FrequencyThresholdedGaussianProcess:
    """
    A Gaussian process with a frequency threshold. The GP is normalized in frequency
    domain to have unit variance. The GP is defined in the original domain,
    but can be recomputed given a new grid. The GP is per channel, so if the input
    has more than one channel, we sample independently from the GP for each channel.
    
    """
    def __init__(self,
                target_freq: float,
                x_channels: int,
                width: int,
                ):
        """
        Args:
        target_freq (float): Target frequency for the GP. This is used to set the lengthscale of the GP.
        x_channels (int): Number of channels in the input. The GP is defined per channel.
        width (int): Width of the GP. This is the number of points in the original domain.
        """

        self.width = width
        self.x_channels = x_channels
        self.target_freq = target_freq
        self.lengthscale = 1.0/(target_freq*2*np.pi)

        # Define mean and covariance of GP
        self.grid = torch.linspace(0,1,self.width)
        self.total_power = self.calculate_total_power()
        self.mean = torch.zeros(self.width)
        self.covar = (1/(2*self.total_power))*self.kernel(self.grid.view(-1,1),self.grid.view(1,-1))  + 1e-5*torch.eye(self.width)

        # Define the base distribution
        self.base_dist = distributions.MultivariateNormal(self.mean,self.covar)
        self.map_lp = self.base_dist.log_prob(self.mean)

    def calculate_total_power(self):
        """
        Calculate the total power of the Gaussian process using the spectral density
        This is used to normalize the samples to be unit variance
        """

        def spectral_density(s):
            return math.sqrt(2*math.pi*self.lengthscale**2) * torch.exp(-s**2 * 2 * self.lengthscale**2 * math.pi**2)
        fft_grid = torch.fft.fftfreq(self.width,1/self.width)
        fft_grid = fft_grid[:self.width//2] # SHOULDN'T NUMBER OF MODES BE WIDTH/2 + 1

        total_power = torch.sum(spectral_density(fft_grid))
        integral = torch.trapz(spectral_density(fft_grid),fft_grid)
        return integral

    def kernel(self,x,y):
        return torch.exp(-(x-y)**2/(2*self.lengthscale**2))
    
    def log_prob(self,x:Tensor):
        # Calculate the log probability of the input x
        lp = torch.zeros(x.shape[0]).to(x.device)
        self.match_device(x.device)
        for i in range(self.x_channels):
            lp += self.base_dist.log_prob(x[:,i])
        return lp.unsqueeze(-1)


    def match_device(self,device):
        # Match the device of the base distribution to the device of the input
        self.base_dist = distributions.MultivariateNormal(self.mean.to(device),self.covar.to(device))
    
    def sample_like(self, x:Tensor, sampling_positions:Optional[Tensor] = None):
        """
        Sample a batch of samples from the GP of the same shape as x.
        
        Args:
        x (Tensor): Input tensor of shape (batch_size, x_channels).
        sampling_positions (Tensor): Optional tensor of shape (n_points,domain_dim) to sample from.
            If None, assume the equispaced original domain.
        """

        batch_size = x.shape[0]

        if sampling_positions is None:
            base_dist = distributions.MultivariateNormal(self.mean.to(x.device),self.covar.to(x.device))
            return self.base_dist.sample((batch_size,self.x_channels))
        elif isinstance(sampling_positions, torch.Tensor):
            sampling_positions = sampling_positions.to(x.device)
            if sampling_positions.ndim == 1: #got (n_points,) so no batch or domain dims
                sampling_positions = sampling_positions.unsqueeze(1)
            
            if sampling_positions.ndim == 2: 
                if sampling_positions.shape[1] != 1:
                    #This means we got (batch_size,n_points)
                    sampling_positions = sampling_positions.unsqueeze(2) #(batch_size,n_points,1)
                else:
                    points_width = sampling_positions.shape[0]
                    mean = torch.zeros(points_width).to(x.device)
                    covar = self.kernel(sampling_positions.view(-1,1),sampling_positions.view(1,-1))/(2*self.total_power)  + 1e-5*torch.eye(points_width).to(x.device)
                    base_dist = distributions.MultivariateNormal(mean,covar)
                    return base_dist.sample((batch_size,self.x_channels))

            if sampling_positions.ndim==3: #got (batch_size,n_points,domain_dim)
                assert sampling_positions.shape[2] == 1 and sampling_positions.shape[0] == batch_size, "sampling_positions must have shape (batch_size,n_points,domain_dim) or (n_points,domain_dim)"
                points_width = sampling_positions.shape[1]
                mean = torch.zeros(batch_size,points_width).to(x.device)
                covar = self.kernel(sampling_positions.view(batch_size,points_width,1),sampling_positions.view(batch_size,1,points_width))/(2*self.total_power)  + 1e-5*torch.eye(points_width).unsqueeze(0).expand(batch_size,-1,-1).to(x.device)
                base_dist = distributions.MultivariateNormal(mean,covar)
                samples = base_dist.sample((self.x_channels,)).view(batch_size,self.x_channels,points_width)
                return samples
        else:
            raise ValueError("sampling_positions must be a tensor of shape (n_points,domain_dim) or None")
    
    def get_pytorch_distribution(self,device="cpu",sampling_positions=None):
        # Get the pytorch distribution for the Gaussian process, useful to pass onto zuko flows.

        if sampling_positions is None:
            dist = Independent(MVN(self.mean.unsqueeze(0).to(device),self.covar.unsqueeze(0).to(device)),1)
        elif isinstance(sampling_positions, torch.Tensor):
            assert sampling_positions.ndim == 2 or sampling_positions.ndim ==1, "sampling_positions must have shape (domain_dim,n_points)"
            if sampling_positions.ndim == 1:
                sampling_positions = sampling_positions.unsqueeze(1)
            mean = torch.zeros(sampling_positions.shape[0]).to(sampling_positions.device)
            covar = (1/(2*self.total_power))*self.kernel(sampling_positions.view(-1,1),sampling_positions.view(1,-1))  + 1e-5*torch.eye(sampling_positions.shape[0]).to(sampling_positions.device)
            dist = Independent(MVN(mean.unsqueeze(0).to(device),covar.unsqueeze(0).to(device)),1)

        else:
            raise ValueError("sampling_positions must be a tensor of shape (domain_dim,n_points) or None")
        return dist


class MatrixNormalWrapper:
    def __init__(self, mean_2d: torch.Tensor, cov_2d: torch.Tensor):
        """
        mean_2d: Tensor of shape [..., H, W] — with optional batch dims
        cov_2d: Tensor of shape [..., H*W, H*W] — batch dims must match mean
        """
        *batch_shape, H, W = mean_2d.shape
        self._event_shape = (H, W)
        self._batch_shape = tuple(batch_shape)

        mean_flat = mean_2d.reshape(*batch_shape, H * W)
        self.dist = MVN(mean_flat, covariance_matrix=cov_2d)

    def sample(self, sample_shape=torch.Size()):
        sample = self.dist.sample(sample_shape)  # shape: sample_shape + batch + [H*W]
        return sample.view(*sample_shape, *self._batch_shape, *self._event_shape)

    def rsample(self, sample_shape=torch.Size()):
        sample = self.dist.rsample(sample_shape)
        return sample.view(*sample_shape, *self._batch_shape, *self._event_shape)

    def log_prob(self, value_2d: torch.Tensor):
        flat_value = value_2d.view(*value_2d.shape[:-2], -1)
        return self.dist.log_prob(flat_value)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def has_rsample(self):
        return self.dist.has_rsample


class FrequencyThresholdedGaussianProcess2d:
    """
    A Gaussian process with a frequency threshold. The GP is normalized in frequency
    domain to have unit variance. The GP is defined in the original domain,
    but can be recomputed given a new grid. The GP is per channel, so if the input
    has more than one channel, we sample independently from the GP for each channel.
    
    """
    def __init__(self,
                target_freq:float,
                x_channels:int,
                domain_size:int,
                default_2d:bool=False,
                ):
        """
        Args:
        target_freq (float): Target frequency for the GP. This is used to set the lengthscale of the GP.
        x_channels (int): Number of channels in the input. The GP is defined per channel.
        domain_size (int): Size of the domain. This is the number of points in the original domain.
        default_2d (bool): If True, the GP is defined in 2D. If False, the GP is defined in 1D.
        """

        # Define base parameters
        self.domain_size = domain_size
        self.x_channels = x_channels
        self.default_2d = default_2d

        # target_freq approximately sets the maximum frequency mode in the GP
        self.target_freq = target_freq
        self.lengthscale = 1.0/(target_freq*2*np.pi)

        # Define mean and covariance of GP
        grid = torch.linspace(0,1,self.domain_size)
        self.x = grid.repeat(self.domain_size)
        self.y = grid.repeat_interleave(self.domain_size)
        self.positions = torch.stack((self.x,self.y),dim=1) # (domain_size**2,2)
        self.mean = torch.zeros(self.domain_size**2)

        # Calculate total power
        self.total_power = self.calculate_total_power() 

        # Specify covariance matrix
        self.covar = (1/(2*self.total_power)) * self.kernel(self.positions)  + 1e-3*torch.eye(self.domain_size**2)  

        if self.default_2d:
            self.base_dist = MatrixNormalWrapper(self.mean.view(1, self.domain_size, self.domain_size), self.covar)
        else:
            self.base_dist = distributions.MultivariateNormal(self.mean, self.covar)

        self.map_lp = self.base_dist.log_prob(self.mean)
        self.chol = None
        self.saved_positions = torch.Tensor([0.0])

    def calculate_total_power(self):
        # Calculate the total power of the Gaussian process using the spectral density
        # This is used to normalize the samples to be unit variance

        def spectral_density(s):
            return math.sqrt(2*math.pi*self.lengthscale**2) * torch.exp(-s**2 * 2 * self.lengthscale**2 * math.pi**2)
            
        # Get frequency components for grid resolution
        n_modes = self.domain_size//2 + 1
        k_base = torch.fft.fftfreq(self.domain_size, 1/self.domain_size)
        k_base = k_base[:n_modes]
        d_k = k_base[1] - k_base[0]
        d_area = d_k**2
        
        # Get all possible combinations of kx and ky for 2D grid 
        kx = k_base.repeat(n_modes)
        ky = k_base.repeat_interleave(n_modes)
        k_abs = torch.sqrt(torch.square(kx) + torch.square(ky))

        # Calculate total power over all frequency components
        total_power = torch.sum(spectral_density(k_abs) * d_area) # APPROXIMATING 2D INTEGRAL HERE 
        return total_power

    # Squared exponential kernel
    def kernel(self, point_positions):
        # Calculate distances between 2D points 
        #point_positions has shape [...,2]
        if point_positions.ndim == 2:
            dist = torch.norm(point_positions[:, None, :] - point_positions[None, :, :], dim=-1)
        elif point_positions.ndim == 3:
            dist = torch.norm(point_positions[:, :, None, :] - point_positions[:, None, :, :], dim=-1)
        
        return torch.exp(-dist**2/(2*self.lengthscale**2))


    def sample_like(self, x: Tensor, sampling_positions: Optional[Tensor]=None):

        batch_size = x.shape[0]
        
        # Sample like a given tensor (take the batch size and number of channels)
        if sampling_positions is None:
            base_dist = distributions.MultivariateNormal(self.mean.to(x.device), self.covar.to(x.device))
            return base_dist.sample((batch_size, self.x_channels)).to(x.device)
            
        # Sampling on specified sampling positions
        elif isinstance(sampling_positions, torch.Tensor):
            sampling_positions = sampling_positions.to(x.device)
            assert sampling_positions.ndim == 2 or sampling_positions.ndim == 3, f"sampling_positions must have shape (n_points, domain_dim) or (batch_size, n_points, domain_dim). Got {sampling_positions.shape}"

            if sampling_positions.ndim == 2:
                if torch.allclose(sampling_positions,self.saved_positions.to(sampling_positions.device)) and sampling_positions.shape == self.saved_positions.shape:
                    z = torch.randn((batch_size, self.x_channels, sampling_positions.shape[0])).to(x.device)
                    samples = torch.einsum("mn, bcn -> bcm", self.chol, z).to(x.device) # (batch_size, x_channels, n_points)
                    return samples
                else:
                    n_points = sampling_positions.shape[0]
                    points_device = sampling_positions.device
                    
                    mean = torch.zeros(n_points).to(points_device)
                    covar = self.kernel(sampling_positions)/(2*self.total_power)  + 1e-3*torch.eye(n_points).to(points_device)
                    self.saved_positions = sampling_positions
                    self.chol = torch.linalg.cholesky(covar)
                    z = torch.randn((batch_size, self.x_channels, n_points)).to(points_device)
                    samples = torch.einsum("mn, bcn -> bcm", self.chol, z).to(x.device) # (batch_size, x_channels, n_points)
                    return samples

            elif sampling_positions.ndim == 3: #got (batch_size,n_points,domain_dim)

                assert sampling_positions.shape[2] == 2 and sampling_positions.shape[0] == batch_size, "sampling_positions must have shape (batch_size,n_points,domain_dim) or (n_points,domain_dim)"
                if torch.allclose(sampling_positions,self.saved_positions.to(sampling_positions.device)) and sampling_positions.shape == self.saved_positions.shape:
                    z = torch.randn((batch_size, self.x_channels, sampling_positions.shape[1])).to(x.device)
                    return torch.einsum("bmn, bcn -> bcm", self.chol, z).to(x.device) # (batch_size, x_channels, n_points)
                else:
                    points_width = sampling_positions.shape[1]
                    mean = torch.zeros(batch_size,points_width).to(x.device)
                    covar = self.kernel(sampling_positions)/(2*self.total_power)  + 1e-3*torch.eye(points_width).unsqueeze(0).expand(batch_size,-1,-1).to(x.device)
                    self.saved_positions = sampling_positions
                    self.chol = torch.linalg.cholesky(covar)
                    z = torch.randn((batch_size, self.x_channels, points_width)).to(x.device)
                    samples = torch.einsum("bmn, bcn -> bcm", self.chol, z).to(x.device) # (batch_size, x_channels, n_points)

                    return samples
                
        else:
            raise ValueError("sampling_positions must be 'original_domain' or a tensor of shape (n_points, domain_dim)")

    
    def get_pytorch_distribution(self, device="cpu", sampling_positions=None, ):
        
        # Get the pytorch distribution for the Gaussian process, useful to pass onto zuko flows.
        if sampling_positions is None:
            if self.default_2d:
                dist = Independent(MatrixNormalWrapper(self.mean.view(1, self.domain_size, self.domain_size).to(device), self.covar.to(device)), 1)
            else:
                dist = Independent(MVN(self.mean.unsqueeze(0).to(device), self.covar.unsqueeze(0).to(device)), 1)
            
        elif isinstance(sampling_positions, torch.Tensor):
            assert sampling_positions.shape[-1] == 2, "sampling_positions must have shape (n_points, domain_dim) or (batch_size, n_points, domain_dim)"

            if sampling_positions.ndim == 2:
            
            
                mean = torch.zeros(sampling_positions.shape[0]).to(sampling_positions.device)
                covar = (1/(2*self.total_power))*self.kernel(sampling_positions)  + 1e-3*torch.eye(sampling_positions.shape[0]).to(sampling_positions.device)
                dist = Independent(MVN(mean.unsqueeze(0).to(device),covar.unsqueeze(0).to(device)),1)

            if sampling_positions.ndim == 3:
                mean = torch.zeros(1,sampling_positions.shape[0],sampling_positions.shape[1]).to(sampling_positions.device)
                n_points = sampling_positions.shape[1]*sampling_positions.shape[0]
                covar = (1/(2*self.total_power))*self.kernel(sampling_positions.view(-1,2))  + 1e-3*torch.eye(n_points).to(sampling_positions.device)
                dist = Independent(MatrixNormalWrapper(mean.to(device), covar), 1)



        else:
            raise ValueError("sampling_positions must be a tensor of shape (n_points,domain_dim) or None")
            
        return dist
    


class WhiteNoise:
    """
    A Gaussian noise base distribution with unit variance and zero mean. 
    
    x_channels: number of input channels
    width: number of points in domain 
    default_2d: if True, the noise is defined in 2D, otherwise in 1D
    """
    def __init__(self, x_channels, width, default_2d = False, **kwargs):

        self.x_channels = x_channels
        self.width = width
        if default_2d is True:
            self.mean = torch.zeros(width, width)
            self.std = torch.ones(width, width)
        else:
            self.mean = torch.zeros(width)
            self.std = torch.ones(width)
        self.base_dist = Normal(0.0, 1.0)  # Standard normal
        self.map_lp = self.base_dist.log_prob(self.mean)
    

    def sample_like(self, x, sampling_positions = None):
        
        batch_size = x.shape[0]
        n_channels = x.shape[1]

        if x.ndim == 3:
            n_points = x.shape[-1]
        elif x.ndim == 4: 
            n_points = x.shape[-2] * x.shape[-1]
        else:
            raise ValueError("x must be three- or four-dimensional.")
        
        mean_tensor = torch.zeros_like(x).to(x.device)
        std_tensor = torch.ones_like(x).to(x.device)

        return torch.normal(mean_tensor, std_tensor).to(x.device)
    
    def get_pytorch_distribution(self, sampling_positions = None, device = "cpu"):

        # Get the pytorch distribution for the white noise, useful to pass to zuko flow.
        reinterpreted_batch_dim = len(self.mean.shape)+1
        dist = Independent(Normal(self.mean.unsqueeze(0).to(device), self.std.unsqueeze(0).to(device)), reinterpreted_batch_ndims=reinterpreted_batch_dim)

        return dist



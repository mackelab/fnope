import warp as wp
import math
import torch
from fnope.simulators.gp_priors import get_gaussian_process_prior_2d


# warp kernels

@wp.kernel
def init_uniform_random_3d(
    array: wp.array3d(dtype=float),
    dim1: int,
    dim2: int,
    min_value: float,
    max_value: float,
    external_seed: int,
):  # pragma: no cover
    """Initialize 2d array with uniform random values

    Parameters
    ----------
    array : wp.array3d
        Array to initialize
    min_value : float
        Min random value
    max_value : float
        Max random value
    external_seed : int
        External seed to use
    """
    b,i,j = wp.tid()
    #    Use the thread index along with external seed to create a unique random state for each thread
    thread_seed = external_seed + b * 1000 + i * 100 + j  # Ensuring uniqueness for each thread

    # Create a random state for this thread
    state = wp.rand_init(thread_seed)
    # Assign a random value to the 3D array
    array[b, i, j] = wp.randf(state, min_value, max_value)

@wp.kernel
def set_values_in_array(array: wp.array3d(dtype=float), values: wp.array3d(dtype=float)):
    """Set each element in the array to the given value."""
    i, j, k = wp.tid()  # Get the thread indices

    # Set the value at the specific location
    array[i, j, k] = values[i, j, k]  # Set the value in the array


class GPPrior:
    def __init__(self, L, lengthscale=10, scale=2,min=-1,max=1):
        """
        Args:
            L: size of the grid
            lengthscale: lengthscale of the GP prior
            scale: scale of the GP prior
            min: min value of the GP prior
            max: max value of the GP prior
        """

        self.L = L
        self.min = min
        self.max = max

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mvn, grid = get_gaussian_process_prior_2d(
            num_points_per_dim=L,
            domain_size=(100.0, 100.0),
            mean=0.0,
            lengthscale=lengthscale,
            sigma=scale,
            device=device,
            # jitter=1e-3
        )

        self.prior_dist = mvn

    def sample(self, size):
        """
        Args:
            size: number of samples to generate
        Returns:
            theta: tensor of shape (size, L, L) 

        """

        theta = self.prior_dist.sample(torch.Size([size])).reshape(
                size, self.L, self.L
            )
        theta = torch.sigmoid(theta) * (self.max - self.min) + self.min 

        return theta

class DarcyGPPrior:
    def __init__(self, n:int, alpha: float = 2.0, tau: float = 3.0, scale:float=1.0, device: str = 'cpu'):
        """
        Spectral sampler for U ~ N(0, (−Δ + τ² I)^(-α)) on an s×s grid over [0,1]^2
        with zero-Neumann boundaries, via the Karhunen–Loève expansion and 2D iDCT.

        Args:
          n      : number of grid points per dimension
          alpha  : exponent on covariance operator
          tau    : shift in precision operator (τ² multiplies I)
          scale  : scaling factor for the field
          device : 'cpu' or 'cuda'
        """
        self.n      = n
        self.alpha  = alpha
        self.tau    = tau
        self.scale = scale
        self.device = torch.device(device)

        # precompute frequency‐domain coefficients
        k = torch.arange(n, device=self.device)
        K1, K2 = torch.meshgrid(k, k, indexing='ij')
        lmbda = (math.pi**2) * (K1**2 + K2**2)      # eigenvalues of −Δ
        # sqrt of covariance eigenvalues: (λ + τ²)^(-α/2)
        self.coef = (lmbda + tau**2).pow(-alpha/2)

        # precompute the DCT basis for iDCT-II (orthonormal)
        arr = torch.arange(n, device=self.device)
        k = arr.clone()
        base = torch.cos(math.pi * (2*arr[:,None] + 1) * k[None,:] / (2*n))
        # normalization factors c[k]
        c = torch.ones(n, device=self.device) * math.sqrt(2/n)
        c[0] = math.sqrt(1/n)
        B = torch.cos(math.pi * k[None, :] * (2*arr[:, None] + 1) / (2*n))
        self.idct_basis = (B * c[None, :])  


    def _idct2(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inverse DCT-II along last two dims, orthonormal.
        X: (..., s, s)
        """
        B = self.idct_basis
        U  = torch.einsum('nk,bkl,ml -> bnm', B,X,B)
        return U
    
    def _dct2(self, U: torch.Tensor) -> torch.Tensor:
        """
        DCT-II along last two dims, orthonormal.
        U: (..., s, s)
        """
        B = self.idct_basis
        L  = torch.einsum('nk,bnm,ml -> bkl', B,U,B)
        return L


    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Draw `num_samples` fields of shape (s, s) each.
        Returns a tensor of shape (num_samples, s, s).
        """
        # draw i.i.d. Gaussian in KL coefficient space
        xi = torch.randn(num_samples, self.n, self.n, device=self.device)
        # scale by sqrt of covariance eigenvalues
        L = self.coef[None, :, :] * xi
        # zero out the constant mode for zero-mean
        L[:, 0, 0] = 0.0
        # inverse DCT² to get the spatial field
        U = self._idct2(L)
        return self.scale*U
    
    def log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: shape (B, s, s)
        returns log p(u) under U ~ N(0, scale^2 * K), K = (−Δ+τ²I)^(-α).
        """
        batch, s1, s2 = u.shape
        assert (s1, s2) == (self.n, self.n)

        # 1) undo overall scale
        u_scaled = u / self.scale

        # 2) go back to KL coefficients L_hat = DCT(u_scaled)
        L_hat = self._dct2(u_scaled)  # shape B x s x s
        L_hat[:, 0, 0] = 0.0           # zero‐mean mode

        # 3) quadratic term
        inv_var = (1.0 / (self.coef**2))[None, :, :]
        quad = (L_hat**2 * inv_var).sum(dim=(1, 2))  # shape (B,)

        # 4) log‐det of covariance in KL:  ∑ij log(coef_ij^2) = 2∑ log coef
        logdetK = 2.0 * torch.log(self.coef).sum()

        # 5) normalization constant
        N = self.n * self.n
        const = N * math.log(2 * math.pi)

        # 6) Jacobian of scaling u→u/scale: contributes −N·log(scale)
        jac = N * math.log(self.scale)

        return -0.5 * quad - 0.5 * logdetK - 0.5 * const - jac
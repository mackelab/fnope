import torch
from torch.distributions import MultivariateNormal


def squared_exponential_kernel(x1, x2, lengthscale, sigma):
    dist = (x1[None, :] - x2[:, None]) ** 2
    return sigma**2 * torch.exp(-0.5 * dist / (lengthscale**2))


def get_gaussian_process_prior_1d(
    num_points, domain_length, mean, lengthscale, sigma, jitter=1e-5, device="cpu"
):
    """
    Return a pytorch MVN defined via a Gaussian Process with a constant mean and
    squared exponential kernel.

    Args:
    num_points: number of points in the prior
    domain_length: length of the domain
    mean: mean of the prior
    lengthscale: lengthscale of the prior
    sigma: standard deviation of the prior
    jitter: jitter to add to the covariance matrix for numerical stability
    """

    mean = torch.full((num_points,), mean).to(device).to(torch.float32)
    x = torch.linspace(0, domain_length, num_points).to(device)
    cov = (
        squared_exponential_kernel(x, x, lengthscale, sigma)
        + torch.eye(num_points).to(device) * jitter
    )
    mvn = MultivariateNormal(mean, covariance_matrix=cov)
    return mvn


def squared_exponential_kernel_2d(x1, x2, lengthscale, sigma):
    """
    Squared exponential kernel for 2D inputs.
    x1, x2: (N, 2) and (M, 2) tensors
    """
    x1 = x1[:, None, :]  # (N, 1, 2)
    x2 = x2[None, :, :]  # (1, M, 2)
    sq_dist = ((x1 - x2) ** 2).sum(-1)  # (N, M)
    return sigma**2 * torch.exp(-0.5 * sq_dist / lengthscale**2)


def get_gaussian_process_prior_2d(
    num_points_per_dim, domain_size, mean, lengthscale, sigma, jitter=1e-4, device="cpu"
):
    """
    Constructs a 2D GP prior using a squared exponential kernel.

    Args:
        num_points_per_dim: int, number of points per spatial dimension (total points = n^2)
        domain_size: float or tuple, size of the 2D domain in each dimension
        mean: float, mean of the GP
        lengthscale: float, kernel lengthscale
        sigma: float, kernel output scale
        jitter: float, added to the diagonal for stability
    """
    if isinstance(domain_size, (int, float)):
        domain_size = (domain_size, domain_size)

    # Estimate grid spacing
    dx = domain_size[0] / num_points_per_dim
    dy = domain_size[1] / num_points_per_dim
    min_spacing = min(dx, dy)

    # Warn or raise if lengthscale is poorly chosen
    if lengthscale < 2 * min_spacing:
        raise ValueError(
            f"Lengthscale {lengthscale} is too small for grid spacing {min_spacing:.4f}. "
            f"Try setting it to at least {2 * min_spacing:.4f}."
        )
    if lengthscale > 2 * max(domain_size):
        raise ValueError(
            f"Lengthscale {lengthscale} is too large for domain size {domain_size}. "
            f"Try a value less than {2 * max(domain_size)}."
        )

    linspace_x = torch.linspace(0, domain_size[0], num_points_per_dim, device=device)
    linspace_y = torch.linspace(0, domain_size[1], num_points_per_dim, device=device)
    grid_x, grid_y = torch.meshgrid(
        linspace_x, linspace_y, indexing="ij"
    )  # shape (n, n)

    # Flatten the 2D grid into a list of coordinates
    grid_points = torch.stack(
        [grid_x.flatten(), grid_y.flatten()], dim=-1
    )  # shape (n^2, 2)

    mean = torch.full((grid_points.shape[0],), mean, dtype=torch.float32, device=device)
    cov = squared_exponential_kernel_2d(grid_points, grid_points, lengthscale, sigma)
    max_diag = cov.diag().max()
    cov += (
        torch.eye(grid_points.shape[0], device=device) * jitter * max_diag
    )  # Add jitter to the diagonal

    mvn = MultivariateNormal(mean, covariance_matrix=cov)
    return mvn, grid_points  # return grid if you want to reshape samples back to 2D

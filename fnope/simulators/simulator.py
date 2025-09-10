import torch
from torch import Tensor
from torchdiffeq import odeint
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def linear_gaussian(
    theta: Tensor,
    likelihood_shift: Tensor,
    likelihood_cov: Tensor,
    num_discarded_dims: int = 0,
) -> Tensor:
    """
    Simulator for linear Gaussian.

    Uses Cholesky decomposition to transform samples from standard Gaussian.

    When `num_discarded_dims>0`, return simulation outputs with as many last dimensions
    discarded. This is implemented by throwing away the last `num_discarded_dims`
    dimensions of theta and then running the linear Gaussian as always.

    Args:
        theta: Parameter sets to be simulated.
        likelihood_shift: The simulator will shift the value of theta by this value.
            Thus, the mean of the Gaussian likelihood will be likelihood_shift+theta.
        likelihood_cov: Covariance matrix of the likelihood.
        num_discarded_dims: Number of output dimensions to discard.

    Returns: Simulated data.
    """
    theta = torch.as_tensor(theta)  # Must be a tensor
    if num_discarded_dims:
        theta = theta[:, :-num_discarded_dims]

    chol_factor = torch.linalg.cholesky(likelihood_cov)

    return likelihood_shift + theta + torch.mm(chol_factor, torch.randn_like(theta).T).T

def gaussian_convolution(theta, kernel_size, kernel_type, likelihood_cov, device):
    """
    Simulator for Gaussian convolution.

    Args:
        theta: Parameter sets to be simulated.
        kernel_size: Size of the convolution kernel.
        kernel_type: Type of the convolution kernel.
        likelihood_cov: Covariance matrix of the likelihood.

    Returns: Simulated data.
    """
    theta = torch.as_tensor(theta)  # Must be a tensor
    if kernel_type == "gaussian":
        kernel = torch.exp(-((torch.arange(kernel_size)-kernel_size/2)/kernel_size).float() ** 2 / 2)
        kernel /= kernel.sum()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    kernel = kernel.to(device)

    x = torch.nn.functional.conv1d(theta.view(theta.shape[0],1,-1), kernel.view(1,1, -1), padding='same')
    x = x.squeeze(1)
    x = x + torch.mm(torch.linalg.cholesky(likelihood_cov), torch.randn_like(theta).to(device).T).T
    return x





def SIR(beta, ts, gamma=0.06,delta = 0.002,likelihood_scale = 0.05,tx=None,I0 = None,device="cpu") -> Tensor:
    if len(beta.shape) == 1:
        beta = beta.unsqueeze(0)
    if isinstance(gamma, torch.Tensor):
        gamma = gamma.to(device)
    if isinstance(delta, torch.Tensor):
        delta = delta.to(device)

    batch_size = beta.shape[0]
    beta = beta.to(device)
    ts = ts.to(device)

    def beta_fn(t,ts, betas):
        index_closest_t = torch.argmin(torch.abs(ts - t))
        if t>ts[-1]:
            return betas[:,-1]
        elif ts[index_closest_t] >= t:
            index_closest_t -= 1
        dt = ts[index_closest_t + 1] - ts[index_closest_t]
        y = (ts[index_closest_t + 1] - t) * betas[:,index_closest_t]/dt + (t - ts[index_closest_t]) * betas[:,index_closest_t + 1]/dt
        return y


    def sir_ode(t,x):

        state ,betas = x
        S = state[:,0]
        I = state[:,1]
        R = state[:,2]
        D = state[:,3]
        beta = beta_fn(t, ts,betas)
        dSdt = -beta * S * I
        dIdt = beta * S * I - (gamma + delta) * I
        dRdt = gamma * I
        dDdt = delta * I
        return torch.stack((dSdt, dIdt, dRdt,dDdt), dim=1), torch.zeros_like(betas)

    if I0 is None:
        initial_I = torch.rand(size=(batch_size,), device=device)*0.2
    elif isinstance(I0, torch.Tensor):
        initial_I = I0.to(device)
        initial_I = initial_I.expand(batch_size)
    elif isinstance(I0, float):
        initial_I = torch.ones(batch_size,device=device) * I0
    initial_S = torch.ones(batch_size,device=device)
    initial_R = torch.zeros(batch_size,device=device)
    initial_D = torch.zeros(batch_size,device=device)
    initial = torch.stack((initial_S, initial_I, initial_R,initial_D), dim=1)


    if tx is not None:
        tx = tx.to(device)
    else:
        tx = ts


    sol, _param = odeint(sir_ode,(initial,beta),tx) #sol is shape (nt, batch_size, 4)
    sol = sol.permute(1,2,0) #(batch_size,4,nt)


    mask = torch.isfinite(sol)
    sol[~mask] = 0.01

    observed = sol[:,[1,2,3],:] #observe infected,recovered,deceased (batch_size,3,nt)


    dist = torch.distributions.LogNormal(loc = torch.log(observed + 1e-6),scale = likelihood_scale*torch.ones_like(observed,device=device))
    noisy_observed = dist.sample()
    return noisy_observed
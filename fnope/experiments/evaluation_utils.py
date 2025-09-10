import torch
from torch import Tensor, zeros, ones
from torch.distributions import Uniform
import pickle
from scipy.stats import kstest, uniform
from sbi.diagnostics.sbc import _run_sbc
from sbi.diagnostics.tarp import _run_tarp, get_tarp_references
from sbi.utils.metrics import l2
from sbi.utils.metrics import c2st
from fnope.utils.metrics import sliced_wasserstein_distance
import warnings
from typing import Dict, Tuple

from fnope.nets.standardizing_net import IdentityStandardizing
from sbi.simulators.linear_gaussian import true_posterior_linear_gaussian_mvn_prior
from fnope.experiments.baseline_sbi_utils import restore_fft, perform_rfft_and_process


import os


# Wrapper classes for posteriors for evaluation metrics
class FNOPosterior:
    def __init__(
        self,
        model,
        theta_shape,
        x_shape,
        point_positions=None,
        ctx_point_positions=None,
        ndims=1,
        sampling_batch_size=1000,
    ):
        self.model = model

        if ndims == 1:
            self.theta_in_channels = theta_shape[0]
            self.theta_space_shape = theta_shape[1:]
            self.x_in_channels = x_shape[0]
            self.x_space_shape = x_shape[1:]

        if ndims ==2:
            self.theta_in_channels = theta_shape[0]
            self.x_in_channels = x_shape[0]
            self.theta_space_shape = theta_shape[1:]
            self.x_space_shape = x_shape[1:]
            
        self.point_positions = point_positions
        self.ctx_point_positions = ctx_point_positions
        assert point_positions is not None, (
            "point_positions must be provided for FNOPosterior"
            "- even if using always equispaced, they will be ignored"
            "but pass the point positions to avoid bugs."
        )
        self.sampling_batch_size = sampling_batch_size

    def sample(self, num_samples, x):
        all_samples = torch.zeros(
            num_samples, *self.theta_space_shape
        ).to(x.device)

        n_full_batches = num_samples // self.sampling_batch_size
        last_batch_size = num_samples % self.sampling_batch_size

        all_theta = []
        all_theta_res = []
        all_x = []

        # Generate all batches and collect
        for i in range(n_full_batches):
            samples = self.model.sample(
                num_samples=self.sampling_batch_size,
                ctx=x.view(self.x_in_channels, *self.x_space_shape),
                point_positions=self.point_positions,
                ctx_point_positions=self.ctx_point_positions,
                atol=1e-2,
                rtol=1e-2,
            ).view(
                self.sampling_batch_size,
                *self.theta_space_shape,
            )
            all_samples[i * self.sampling_batch_size : (i + 1) * self.sampling_batch_size] = samples

        if last_batch_size > 0:
            # Generate last batch
            samples = self.model.sample(
                num_samples=last_batch_size,
                ctx=x.view(self.x_in_channels, *self.x_space_shape),
                point_positions=self.point_positions,
                ctx_point_positions=self.ctx_point_positions,
                atol=1e-2,
                rtol=1e-2,
            ).view(
                last_batch_size,
                *self.theta_space_shape,
            )
            all_samples[-last_batch_size:] = samples


        return all_samples

    def unnormalized_log_prob(self, theta, x):
        # compute unnormalized log prob
        log_probs = self.model.unnormalized_log_prob(
            theta.view(-1, self.theta_in_channels, *self.theta_space_shape),
            ctx=x.view(-1, self.x_in_channels, *self.x_space_shape),
            point_positions=self.point_positions,
            ctx_point_positions=self.ctx_point_positions,
            atol=1e-2,
            rtol=1e-2,
        ).view(-1,)
        return log_probs


# Wrapper class for ground truth posterior (only for linear gaussian)
class GTPosterior:
    def __init__(self, likelihood_shift, likelihood_cov, gp_prior, device="cpu"):
        self.likelihood_shift = likelihood_shift
        self.likelihood_cov = likelihood_cov
        self.loc = gp_prior.mean
        self.covariance_matrix = gp_prior.covariance_matrix

        self.device = device

    def sample(self, num_samples, x):
        # sample from posterior
        return (
            true_posterior_linear_gaussian_mvn_prior(
                x_o=x.view(-1, x.shape[-1]).to(self.device).to(torch.float64),
                likelihood_shift=self.likelihood_shift.to(torch.float64),
                likelihood_cov=self.likelihood_cov.to(torch.float64),
                prior_mean=self.loc.to(self.device).to(torch.float64),
                prior_cov=self.covariance_matrix.to(self.device).to(torch.float64),
            )
            .sample((num_samples,))
            .to(torch.float32)
        )


# Wrapper class for SBI-based posteriors
class SBIPosterior:
    def __init__(
        self,
        posterior,
        data_representation,
        theta_shape,
        x_shape,
        theta_standardizing_net=IdentityStandardizing(),
        theta_pad_width=None,
        x_dims=1,
        theta_dims=1,
    ):
        """Posterior wrapper for SBI-based posteriors.
        Args:
            posterior: SBI posterior object
            data_representation: data representation type, either "raw" or "fourier"
            theta_shape:
            x_shape:
            theta_standardizing_net: standardizing net for theta
            theta_pad_width: padding width for fourier representation
            x_dims: number of dimensions for x in the original space
            theta_dims: number of dimensions for theta in the original space
        """
        assert data_representation in [
            "raw",
            "fourier",
        ], "data_representation must be either 'raw' or 'fourier'"
        self.data_representation = data_representation
        if self.data_representation == "fourier":
            assert (
                theta_pad_width is not None
            ), "theta_pad_width must be provided for fourier data representation"
        self.theta_standardizing_net = theta_standardizing_net
        self.posterior = posterior
        if theta_dims == 1:
            self.theta_in_channel = theta_shape[0]
            self.theta_space_shape = theta_shape[1:]
        elif theta_dims == 2:
            self.theta_in_channel = theta_shape[0]
            self.theta_space_shape = theta_shape[1:]
        if x_dims == 1:
            self.x_in_channels = x_shape[0]
            self.x_space_shape = x_shape[1:]
        elif x_dims == 2:
            self.x_in_channels = x_shape[0]
            self.x_space_shape = x_shape[1:]
        
        self.theta_pad_width = theta_pad_width
        self.theta_dims = theta_dims
        self.x_dims = x_dims

    def sample(self, num_samples, x):
        self.posterior.set_default_x(
            x.view(
                self.x_space_shape,
            )
        )
        if self.data_representation == "fourier":
            # sample from posterior
            freq_samples = self.posterior.sample(
                (num_samples,), show_progress_bars=False
            )
            freq_samples = self.theta_standardizing_net.unstandardize(freq_samples)

            if self.theta_dims == 1:
                real_samples = restore_fft(
                    freq_samples,
                    size=self.theta_space_shape,
                    pad_width=self.theta_pad_width,
                    originial_dims=1,
                )
            elif self.theta_dims == 2:
                real_samples = restore_fft(
                    freq_samples,
                    size=self.theta_space_shape,
                    pad_width=self.theta_pad_width,
                    originial_dims=2,
                )

        elif self.data_representation == "raw":
            real_samples = self.posterior.sample(
                (num_samples,), show_progress_bars=False
            ).reshape(
                num_samples,
                *self.theta_space_shape,
            )
            real_samples = self.theta_standardizing_net.unstandardize(
                real_samples.unsqueeze(1), point_positions=None
            ).squeeze(
                1
            )  # Filter Standardizing needs channel dimension

        return real_samples
    
    def log_prob(self, theta, x,n_fft_modes=None):
        # compute unnormalized log prob
        if self.data_representation == "fourier":
            # standardize the theta samples
            freq_samples, (H_pad, W_pad) = perform_rfft_and_process(
                theta,
                n_fft_modes,
                pad_width=self.theta_pad_width,
            )
            freq_samples = self.theta_standardizing_net.standardize(freq_samples)
            # compute unnormalized log prob
            log_probs = self.posterior.log_prob_batched(
                freq_samples.unsqueeze(0).to(x.device),
                x,
            ).view(-1,)
        elif self.data_representation == "raw":
            raw_samples = self.theta_standardizing_net.standardize(
                theta.view(-1, *self.theta_space_shape),
                point_positions=None,
            ).view(-1, *self.theta_space_shape)
            log_probs = []
            for i in range(raw_samples.shape[0]):
                self.posterior.set_default_x(
                    x[i].view(
                        self.x_space_shape,
                    )
                )
                lp = self.posterior.log_prob(
                    raw_samples[i].view(1,-1).to(x.device),
                    ode_kwargs = {"exact": False},
                ).item()
                log_probs.append(lp)
            log_probs = torch.tensor(log_probs, device=theta.device)
        return log_probs

    def to(self, device):
        self.theta_standardizing_net.to(device)
        return self


def check_uniformity_frequentist(ranks, num_posterior_samples) -> Tensor:
    """Return p-values for uniformity of the ranks.

    Calculates Kolomogorov-Smirnov test using scipy.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.

    Returns:
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
    """
    kstest_pvals = torch.tensor(
        [
            kstest(rks, uniform(loc=0, scale=num_posterior_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=torch.float32,
    )

    return kstest_pvals


def check_uniformity_c2st(
    ranks, num_posterior_samples, num_repetitions: int = 1
) -> Tensor:
    """Return c2st scores for uniformity of the ranks.

    Run a c2st between ranks and uniform samples.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_repetitions: repetitions of C2ST tests estimate classifier variance.

    Returns:
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
    """

    c2st_scores = torch.tensor(
        [
            [
                c2st(
                    rks.unsqueeze(1),
                    Uniform(zeros(1), num_posterior_samples * ones(1)).sample(
                        torch.Size((ranks.shape[0],))
                    ),
                )
                for rks in ranks.T
            ]
            for _ in range(num_repetitions)
        ]
    )

    # Use variance over repetitions to estimate robustness of c2st.
    c2st_std = c2st_scores.std(0, correction=0 if num_repetitions == 1 else 1)
    if (c2st_std > 0.05).any():
        warnings.warn(
            f"C2ST score variability is larger than {0.05}: std={c2st_scores.std(0)}, "
            "result may be unreliable. Consider increasing the number of samples.",
            stacklevel=2,
        )

    # Return the mean over repetitions as c2st score estimate.
    return c2st_scores.mean(0)


def check_sbc(
    ranks: Tensor,
    prior_samples: Tensor,
    dap_samples: Tensor,
    num_posterior_samples: int = 1000,
    num_c2st_repetitions: int = 1,
) -> Dict[str, Tensor]:
    """Return uniformity checks and data averaged posterior checks for SBC.
    We override the default check_sbc function from `sbi` which also calculates
    the c2st between the data-averaged posterior and the prior samples, which
    is very slow for high-dimensional posteriors!

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        prior_samples: N samples from the prior
        dap_samples: N samples from the data averaged posterior
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_c2st_repetitions: number of times c2st is repeated to estimate robustness.

    Returns (all in a dictionary):
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
    """
    if ranks.shape[0] < 100:
        warnings.warn(
            "You are computing SBC checks with less than 100 samples. These checks"
            " should be based on a large number of test samples theta_o, x_o. We"
            " recommend using at least 100.",
            stacklevel=2,
        )

    ks_pvals = check_uniformity_frequentist(ranks, num_posterior_samples)
    c2st_ranks = check_uniformity_c2st(
        ranks, num_posterior_samples, num_repetitions=num_c2st_repetitions
    )

    return dict(
        ks_pvals=ks_pvals,
        c2st_ranks=c2st_ranks,
    )


@torch.no_grad()
def run_sbc_save_results(
    theta,
    x,
    posterior,
    num_bins=30,
    num_posterior_samples=1000,
    downsampling_scale=1,
    path_to_save="",
    device="cpu",
):
    num_sbc_samples = theta.shape[0]
    theta = theta.view(theta.shape[0], -1).to(device)

    reduce_fns = [
        eval(f"lambda theta, x: theta[:, {downsampling_scale*i}]")
        for i in range(theta.shape[-1] // downsampling_scale)
    ]


    # get posterior samples
    # of shape (num_samples, batch_size, dim_parameters).
    posterior_samples = torch.zeros(
        num_posterior_samples, num_sbc_samples, theta.shape[-1]
    ).to(device)

    for i in range(num_sbc_samples):
        print(f"Sample {i}")
        # sample from posterior
        posterior_samples[:, i, :] = posterior.sample(
            num_posterior_samples, x[i]
        ).reshape(num_posterior_samples,-1)

    # data average posterior samples
    dap_samples = posterior_samples[0, :, :]
    assert dap_samples.shape == (
        num_sbc_samples,
        theta.shape[-1],
    ), "Wrong dap shape."

    # posterior is not needed in _run_sbc:

    ranks = _run_sbc(
        theta,
        x,
        posterior_samples,
        posterior=None,
        reduce_fns=reduce_fns,
    )

    check_stats = check_sbc(
        ranks,
        theta.view(num_sbc_samples, -1),
        dap_samples,
        num_posterior_samples=num_posterior_samples,
    )

    coverage_values = ranks / num_posterior_samples

    # coverage_values = coverage_values.view(-1,)
    print("coverage values shape: ", coverage_values.shape)

    atcs = []
    absolute_atcs = []
    # In principle doable with torch.histrogramdd() but this is bugged right now.
    for dim_idx in range(coverage_values.shape[1]):
        # calculate empirical CDF via cumsum and normalize
        hist, alpha_grid = torch.histogram(
            coverage_values[:, dim_idx], density=True, bins=num_bins
        )
        # add 0 to the beginning of the ecp curve to match the alpha grid
        ecp = torch.cat([Tensor([0]), torch.cumsum(hist, dim=0) / hist.sum()])
        atc = (ecp - alpha_grid).mean().item()
        absolute_atc = (ecp - alpha_grid).abs().mean().item()
        atcs.append(atc)
        absolute_atcs.append(absolute_atc)

    atcs = torch.tensor(atcs)
    absolute_atcs = torch.tensor(absolute_atcs)
    print("atcs: ", atcs)
    print("absolute_atcs: ", absolute_atcs)

    # construct dict to save the evaluation results
    sbc_results = {
        "ranks": ranks,
        "check_stats": check_stats,
        "atcs": atcs,
        "absolute_atcs": absolute_atcs,
    }

    # save the results
    with open(os.path.join(path_to_save, "sbc_results.pkl"), "wb") as f:
        pickle.dump(sbc_results, f)

    return sbc_results


def check_tarp(
    ecp: Tensor,
    alpha: Tensor,
) -> Tuple[float, float, float]:
    r"""check the obtained TARP credibitlity levels and
    expected coverage probabilities. This will help to uncover underdispersed,
    well covering or overdispersed posteriors.

    Args:
        ecp: expected coverage probabilities computed with the TARP method,
            i.e. first output of ``run_tarp``.
        alpha: credibility levels $\alpha$, i.e. second output of ``run_tarp``.

    Returns:
        atc: area to curve, the difference between the ecp and alpha curve for
            alpha values larger than 0.5. This number should be close to ``0``.
            Values larger than ``0`` indicated overdispersed distributions (i.e.
            the estimated posterior is too wide). Values smaller than ``0``
            indicate underdispersed distributions (i.e. the estimated posterior
            is too narrow). Note, this property of the ecp curve can also
            indicate if the posterior is biased, see figure 2 of the paper for
            details (https://arxiv.org/abs/2302.03026).
        ks prob: p-value for a two sample Kolmogorov-Smirnov test. The null
             hypothesis of this test is that the two distributions (ecp and
             alpha) are identical, i.e. are produced by one common CDF. If they
             were, the p-value should be close to ``1``. Commonly, people reject
             the null if p-value is below 0.05!
    """

    # area to curve: difference between ecp and alpha above 0.5.
    atc = (ecp - alpha).mean().item()
    absolute_atc = (ecp - alpha).abs().mean().item()

    # Kolmogorov-Smirnov test between ecp and alpha
    kstest_pvals: float = kstest(ecp.numpy(), alpha.numpy())[1]  # type: ignore

    return atc, absolute_atc, kstest_pvals


@torch.no_grad()
def run_tarp_save_results(
    theta,
    x,
    posterior,
    reference_points,
    num_posterior_samples=1000,
    path_to_save="",
    device="cpu",
):

    num_sbc_samples = theta.shape[0]
    theta = theta.view(theta.shape[0], -1).to(device)

    # for fmpe we need to sample ourself,
    # as the model is not an SBI object
    num_tarp_samples = theta.shape[0]

    # get posterior samples
    # of shape (num_samples, batch_size, dim_parameters).
    posterior_samples = torch.zeros(
        num_posterior_samples, num_tarp_samples, theta.shape[-1]
    ).to(device)

    for i in range(num_tarp_samples):
        print(f"Sample {i}")
        # sample from posterior
        posterior_samples[:, i, :] = posterior.sample(
            num_posterior_samples, x[i]
        ).reshape(num_posterior_samples, -1)

    if reference_points is None:
        reference_points = get_tarp_references(theta.view(num_tarp_samples, -1)).to(
            device
        )

    # posterior is not needed in _run_tarp:
    ecp, alpha_grid = _run_tarp(
        posterior_samples,
        theta.view(num_tarp_samples, -1).to(device),
        references=reference_points.to(device),
        distance=l2,
        z_score_theta=True,
    )

    atc, absolute_atc, kstest_pvals = check_tarp(ecp, alpha_grid)

    # construct dict to save the evaluation results
    tarp_results = {
        "ecp": ecp,
        "alpha_grid": alpha_grid,
        "absolute_atcs": absolute_atc,
        "atcs": atc,
        "kstest_pvals": kstest_pvals,
    }

    # save the results
    with open(os.path.join(path_to_save, "tarp_results.pkl"), "wb") as f:
        pickle.dump(tarp_results, f)

    return tarp_results


@torch.no_grad()
def run_swd_save_results(
    x,
    posterior1,
    posterior2,
    num_posterior_samples=1000,
    path_to_save="",
    mode="npe",  # "npe" or "fmpe"
    device="cpu",
):
    swds = []

    for i in range(x.shape[0]):
        print(f"Sample {i}")
        # sample from posterior
        posterior_samples1 = posterior1.sample(
            num_posterior_samples, x[i].view(1, -1)
        ).view(num_posterior_samples, -1)
        posterior_samples2 = posterior2.sample(
            num_posterior_samples, x[i].view(1, -1)
        ).view(num_posterior_samples, -1)

        # compute sliced wasserstein distance
        swd = sliced_wasserstein_distance(
            posterior_samples1, posterior_samples2, device=device
        ).item()
        swds.append(swd)
    swds = torch.tensor(swds)
    print("swds")
    print(swds)
    # construct dict to save the evaluation results
    swd_results = {
        "swd": swds,
    }
    # save the results
    with open(os.path.join(path_to_save, "swd_results.pkl"), "wb") as f:
        pickle.dump(swd_results, f)

    return swd_results


def run_predictive_checks_save_results(
    x,
    predictive_samples,
    path_to_save="",
    prependix="",
):
    mses = []
    sds = []

    for i in range(x.shape[0]):
        print(f"Observation {i}")

        predictives = predictive_samples[:, i].view(-1, *x[i].shape)
        # compute standard deviations
        sd = predictives.std(dim=0).view(1, *x[i].shape)
        sds.append(sd)
        # Compute MSEs
        error = ((x[i].view(1, *x[i].shape) - predictives) ** 2)
        mse = error.view(error.shape[0], -1).mean(dim=1).view(-1)
        mses.append(mse)
    mses = torch.stack(mses)
    sds = torch.cat(sds, dim=0)
    print("sds: ", sds.shape)
    print("mses: ", mses)
    predictive_check_results = {
        "mses": mses,
        "sds": sds,
    }
    # save the results
    with open(os.path.join(path_to_save, prependix+"predictive_check_results.pkl"), "wb") as f:
        pickle.dump(predictive_check_results, f)
    return predictive_check_results


def compute_power(x):
    """ Compute the power spectrum of a 1D signal.

    It assumses that the total timescale is 1.0 

    Args:
        x (torch.Tensor): Input tensor of shape (n_samples, seq_len).
    Returns:
        power_spectrum (torch.Tensor): Mean power spectrum of the input signal
            (only positive frequencies).
        freqs (torch.Tensor): Frequencies corresponding to the power spectrum.
    """
    if x.ndim != 2:
        raise ValueError("Input tensor x must be of shape (n_samples, seq_len)")


    seq_len = x.shape[1]
    T = 1.0  # total timescale
    dt = T / seq_len  # sample spacing

    freqs = torch.fft.fftfreq(seq_len, d=dt)

    x_fft = torch.fft.fft(x, dim=-1)
    power_spectrum = torch.abs(x_fft) ** 2

    # return only the positive frequencies
    positive_freqs = freqs >= 0
    freqs = freqs[positive_freqs]
    power_spectrum = power_spectrum[..., positive_freqs]

    return torch.mean(power_spectrum, dim=0), freqs
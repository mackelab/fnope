import torch
import numpy as np
from typing import Tuple, List, Dict, Union
from fnope.flow_matching.fnope_1D import FNOPE_1D
from fnope.flow_matching.fnope_2D import FNOPE_2D


def rejection_sample(
    num_samples: int,
    bounds_cont: List[Tuple[float, float]],
    bounds_fin: List[Tuple[float, float]],
    proposal: Union[FNOPE_1D, FNOPE_2D],
    proposal_kwargs: Dict,
    sampling_batch_size: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rejection sampling from a proposal distribution.

    Args:
        num_samples (int): Number of samples to draw.
        bounds_cont (List[Tuple[float, float]]): Bounds for each dimension.
        bounds_fin (List[Tuple[float, float]]): Bounds for each dimension.
        proposal (Union[FNOPE_1D, FNOPE_2D]): Proposal distribution.
        proposal_kwargs (Dict): Additional arguments for the proposal distribution.
        sampling_batch_size (int): Batch size for sampling.

    Returns:
        torch.Tensor: Accepted samples.
    """
    # Initialize an empty list to store the accepted samples
    accepted_samples_cont = []
    accepted_samples_fin = []

    # Loop until we have enough accepted samples
    while len(accepted_samples_cont) < num_samples:
        # Sample from the proposal distribution
        samples_cont,samples_fin = proposal.sample(sampling_batch_size, **proposal_kwargs)
        samples_cont = samples_cont.detach().cpu().numpy()
        samples_fin = samples_fin.detach().cpu().numpy()

        # Check if the samples are within the bounds
        in_bounds_cont = np.all(
            [np.logical_and(samples_cont[:, 0, i] >= bounds_cont[i][0], samples_cont[:, 0, i] <= bounds_cont[i][1]) for i in range(len(bounds_cont))],
            axis=0,
        )
        in_bounds_fin = np.all(
            [np.logical_and(samples_fin[:, i] >= bounds_fin[i][0], samples_fin[:, i] <= bounds_fin[i][1]) for i in range(len(bounds_fin))],
            axis=0,
        )
        in_bounds = np.logical_and(in_bounds_cont, in_bounds_fin)

        # Accept the samples that are within the bounds
        accepted_samples_cont.extend(samples_cont[in_bounds])
        accepted_samples_fin.extend(samples_fin[in_bounds])

    return torch.from_numpy(np.array(accepted_samples_cont[:num_samples])), torch.from_numpy(np.array(accepted_samples_fin[:num_samples]))


    
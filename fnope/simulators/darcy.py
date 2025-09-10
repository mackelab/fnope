# this code is adapted from the physics nemo datapipe for darcy flow:
# https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/datapipes/benchmarks/darcy.py

import warp as wp
import torch
from typing import Dict, Tuple, Union


from physicsnemo.datapipes.benchmarks.kernels.finite_difference import (
    darcy_mgrid_jacobi_iterative_batched_2d,
    mgrid_inf_residual_batched_2d,
)
from physicsnemo.datapipes.benchmarks.kernels.utils import bilinear_upsample_batched_2d

import copy

from fnope.simulators.darcy_utils import (GPPrior, DarcyGPPrior, set_values_in_array)

class Darcy2D():
    """
    Darcy2D is a datapipe for the Darcy flow problem in 2D.

    Args:
        resolution (int): Resolution of the grid.
        batch_size (int): Batch size.
        nr_permeability_freq (int): Number of frequencies for the permeability field.
        normaliser (Dict[str, Tuple[float, float]]): Normalisation factors for the input and output fields.
        convergence_threshold (float): Convergence threshold for the iterative solver.

    Returns:
        None
    """
    def __init__(
        self,
        batch_size: int = 8,
        nr_permeability_freq: int = 2,
        resolution: int = 32, # resolution n -> parameters sampled on (n+1)x(n+1) grid, observation return on n x n grid
        max_iterations: int = 30000,
        convergence_threshold: float = 1e-6,
        iterations_per_convergence_check: int = 1000,
        nr_multigrids: int = 4,
        normaliser: Union[Dict[str, Tuple[float, float]], None] = None,
        snr: float = 100.0,
        device: Union[str, torch.device] = "cuda",
        prior: str = "GP",
        prior_params: Dict[str, float] = {"scale": 2.0, 
                                          "lengthscale": 10.0, 
                                          "min_permeability": 0.5, 
                                          "max_permeability": 2.0},
        prior_params_darcy: Dict[str, float] = {"tau": 9.0,
                                                "alpha": 2.0,
                                                "scale": 2000.0,
                                                },

    ):
        
        self.batch_size = batch_size
        self.nr_permeability_freq = nr_permeability_freq
        # simulation params
        self.resolution = resolution
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.iterations_per_convergence_check = iterations_per_convergence_check
        self.nr_multigrids = nr_multigrids
        self.normaliser = normaliser
        self.snr = snr

        # check normaliser keys
        if self.normaliser is not None:
            if not {"permeability", "darcy"}.issubset(set(self.normaliser.keys())):
                raise ValueError(
                    "normaliser need to have keys permeability and darcy with mean and std"
                )

        # Set up device for warp, warp has same naming convention as torch.
        if isinstance(device, torch.device):
            device = str(device)
        self.device = device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            raise Warning("CUDA is not available. Please run on a machine with CUDA support.")

        # spatial dims
        self.dx = 1.0 / (self.resolution + 1)  # pad edges by 1 for multi-grid
        self.dim = (self.batch_size, self.resolution + 1, self.resolution + 1)
        self.fourier_dim = (
            4,
            self.batch_size,
            self.nr_permeability_freq,
            self.nr_permeability_freq,
        )

        # assert resolution is compatible with multi-grid method
        if (resolution % 2 ** (nr_multigrids - 1)) != 0:
            raise ValueError("Resolution is incompatible with number of sub grids.")

        # allocate arrays for constructing dataset
        self.darcy0 = wp.zeros(self.dim, dtype=float, device=self.device)
        self.darcy1 = wp.zeros(self.dim, dtype=float, device=self.device)
        self.permeability = wp.zeros(self.dim, dtype=float, device=self.device)
        self.rand_fourier = wp.zeros(self.fourier_dim, dtype=float, device=self.device)
        self.inf_residual = wp.zeros([1], dtype=float, device=self.device)

        # Output tenors
        self.output_k = None
        self.output_p = None

        # set up prior
        if prior == "GP":     
            self.prior = GPPrior(
                L=self.dim[1],
                lengthscale=prior_params["lengthscale"],
                scale=prior_params["scale"],
                min=prior_params["min_permeability"],
                max=prior_params["max_permeability"],
            )
        elif prior == "Darcy_GP":
            self.prior = DarcyGPPrior(
                n=self.dim[1],
                tau=prior_params_darcy["tau"],
                alpha=prior_params_darcy["alpha"],
                scale=prior_params_darcy["scale"],
                device =self.device,
            )
        else:
            raise ValueError("Prior not implemented")

    def _sample_prior(self,) -> torch.Tensor:
        thetas = self.prior.sample(self.batch_size)
        return thetas

    def initialize_batch(self, initial_state) -> None:
        """Initializes arrays for new batch of simulations"""
        initial_state = initial_state.to(self.device)
        initial_state_wp = wp.from_torch(initial_state)
        assert initial_state_wp.shape == self.permeability.shape, "initial state shape mismatch. Initial_state_wp shape: {}, permeability shape: {}".format(initial_state_wp.shape, self.permeability.shape)

        wp.launch(
                kernel=set_values_in_array,
                dim=self.dim,
                inputs=[self.permeability,initial_state_wp],
                device=self.device,
            )
        
        if False:
            # initialize permeability with random states
            self.permeability.zero_()
            seed = np.random.randint(np.iinfo(np.uint64).max, dtype=np.uint64)
            print("seed: ", seed)
            wp.launch(
                kernel=init_uniform_random_3d,
                dim=self.dim,
                inputs=[self.permeability,self.dim[1],self.dim[2], -1.0, 1.0, seed],
                device=self.device,
            )
            
        # zero darcy arrays
        self.darcy0.zero_()
        self.darcy1.zero_()

    def solve_batch(self,initial_state) -> None:
        """Solve for new batch of simulations"""

        # initialize tensors with random permeability
        self.initialize_batch(initial_state)

        # run solver
        for res in range(self.nr_multigrids):
            # calculate grid reduction factor and reduced dim
            grid_reduction_factor = 2 ** (self.nr_multigrids - res - 1)
            if grid_reduction_factor > 1:
                multigrid_dim = tuple(
                    [self.batch_size] + 2 * [(self.resolution) // grid_reduction_factor]
                )
            else:
                multigrid_dim = self.dim

            # run till max steps is reached
            for k in range(
                self.max_iterations // self.iterations_per_convergence_check
            ):
                # run jacobi iterations
                for s in range(self.iterations_per_convergence_check):
                    # iterate solver
                    wp.launch(
                        kernel=darcy_mgrid_jacobi_iterative_batched_2d,
                        dim=multigrid_dim,
                        inputs=[
                            self.darcy0,
                            self.darcy1,
                            self.permeability,
                            1.0,
                            self.dim[1],
                            self.dim[2],
                            self.dx,
                            grid_reduction_factor,
                        ],
                        device=self.device,
                    )

                    # swap buffers
                    (self.darcy0, self.darcy1) = (self.darcy1, self.darcy0)

                # compute residual
                self.inf_residual.zero_()
                wp.launch(
                    kernel=mgrid_inf_residual_batched_2d,
                    dim=multigrid_dim,
                    inputs=[
                        self.darcy0,
                        self.darcy1,
                        self.inf_residual,
                        grid_reduction_factor,
                    ],
                    device=self.device,
                )
                normalized_inf_residual = self.inf_residual.numpy()[0]

                # check if converged
                if normalized_inf_residual < (
                    self.convergence_threshold * grid_reduction_factor
                ):
                    break

            # upsample to higher resolution
            if grid_reduction_factor > 1:
                wp.launch(
                    kernel=bilinear_upsample_batched_2d,
                    dim=self.dim,
                    inputs=[
                        self.darcy0,
                        self.dim[1],
                        self.dim[2],
                        grid_reduction_factor,
                    ],
                    device=self.device,
                )


    def sample_darcy(self):

        initial_state = self._sample_prior()
        exp_initial_state = torch.exp(initial_state)
        self.solve_batch(exp_initial_state)

        # convert warp arrays to pytorch
        permeability = wp.to_torch(self.permeability)
        darcy = wp.to_torch(self.darcy0)

        # add channel dims
        permeability = torch.unsqueeze(permeability, axis=1)
        darcy = torch.unsqueeze(darcy, axis=1)



        # normalize values
        if self.normaliser is not None:
            permeability = (
                permeability - self.normaliser["permeability"][0]
            ) / self.normaliser["permeability"][1]
            darcy = (darcy - self.normaliser["darcy"][0]) / self.normaliser[
                "darcy"
            ][1]

        # add noise
        maxval = darcy.max()
        darcy = darcy/maxval
        diagonal_variances = (darcy.pow(2).mean(dim=(1,2,3))).view(-1,1,1,1) / self.snr
        noise = torch.randn_like(darcy) * torch.sqrt(diagonal_variances)
        darcy = darcy + noise
        darcy *= maxval
        # crop edges by 1 from multi-grid (messy)
        permeability_res = permeability[:, :, : self.resolution, : self.resolution]
        darcy_res = darcy[:, :, : self.resolution, : self.resolution]

        # CUDA graphs static copies
        if self.output_k is None:
            self.output_k = permeability_res
            self.output_p = darcy_res
        else:
            self.output_k.data.copy_(permeability_res)
            self.output_p.data.copy_(darcy_res)

        #return {"permeability": copy.deepcopy(self.output_k), "darcy": copy.deepcopy(self.output_p)}
                #return {"permeability": copy.deepcopy(self.output_k), "darcy": copy.deepcopy(self.output_p)}
        return copy.deepcopy(initial_state),copy.deepcopy(self.output_k).squeeze(), copy.deepcopy(darcy).squeeze()
    
    def simulate_darcy(self, initial_state):
        """Simulate Darcy flow with given initial state"""
        exp_initial_state = torch.exp(initial_state)
        self.solve_batch(exp_initial_state)
        # convert warp arrays to pytorch
        permeability = wp.to_torch(self.permeability)
        darcy = wp.to_torch(self.darcy0)

        # add channel dims
        permeability = torch.unsqueeze(permeability, axis=1)
        darcy = torch.unsqueeze(darcy, axis=1)


        # normalize values
        if self.normaliser is not None:
            permeability = (
                permeability - self.normaliser["permeability"][0]
            ) / self.normaliser["permeability"][1]
            darcy = (darcy - self.normaliser["darcy"][0]) / self.normaliser[
                "darcy"
            ][1]

        # add noise
        maxval = darcy.max()
        darcy = darcy/maxval
        diagonal_variances = (darcy.pow(2).mean(dim=(1,2,3))).view(-1,1,1,1) / self.snr
        noise = torch.randn_like(darcy) * torch.sqrt(diagonal_variances)
        darcy = darcy + noise
        darcy *= maxval
        # crop edges by 1 from multi-grid (messy)
        permeability_res = permeability[:, :, : self.resolution, : self.resolution]
        darcy_res = darcy[:, :, : self.resolution, : self.resolution]

        # CUDA graphs static copies
        if self.output_k is None:
            self.output_k = permeability_res
            self.output_p = darcy_res
        else:
            self.output_k.data.copy_(permeability_res)
            self.output_p.data.copy_(darcy_res)

        return copy.deepcopy(initial_state),copy.deepcopy(self.output_k).squeeze(), copy.deepcopy(darcy).squeeze()





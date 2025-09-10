import torch
from omegaconf import DictConfig
from fnope.simulators.darcy import Darcy2D
from fnope.utils.misc import get_data_dir, set_seed
from pathlib import Path


import pickle
import hydra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="data_generation"
)
def sample_darcy(cfg: DictConfig):

    # get hyperparameters
    n_samples = cfg.data_config.N
    resolution = cfg.data_config.resolution
    seed = cfg.data_config.seed
    batch_size = cfg.data_config.batch_size
    prior_mode = cfg.data_config.prior_mode

    # put together prior info
    prior_params = {
        "scale": cfg.data_config.prior_variance,
        "lengthscale": cfg.data_config.prior_lengthscale,
        "min_permeability": cfg.data_config.prior_min,
        "max_permeability": cfg.data_config.prior_max,
    }

    prior_params_darcy_gp = {
        "tau": cfg.data_config.tau,
        "alpha": cfg.data_config.alpha,
        "scale": cfg.data_config.prior_scale,
    }

    data_dir = get_data_dir()
    seed = cfg.data_config.seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up simulator
    darcy = Darcy2D(
        prior=prior_mode,
        prior_params=prior_params,
        prior_params_darcy=prior_params_darcy_gp,
        batch_size=batch_size,
        resolution=resolution,
        snr=cfg.data_config.snr,
    )

    # batch this to n at a time
    batch_size = batch_size
    n_full_batches = n_samples // batch_size
    last_batch_size = n_samples % batch_size

    all_theta = []
    all_x = []

    # Generate all batches and collect
    for i in range(n_full_batches):
        theta_temp, theta_res, x_temp = darcy.sample_darcy()
        all_x.append(x_temp.cpu())
        all_theta.append(theta_temp.cpu()) # theta is the parameter on the grid we sample the prior from.
        print(f"batch {i+1}/{n_full_batches} done.")

    if last_batch_size > 0:
        # Generate last batch
        theta_temp, theta_res, x_temp = darcy.sample_darcy()
        all_x.append(x_temp.cpu()[:last_batch_size])
        all_theta.append(theta_temp.cpu()[:last_batch_size])

    # Concatenate all
    all_theta = torch.cat(all_theta, dim=0)
    assert all_theta.shape[0] == n_samples
    all_x = torch.cat(all_x, dim=0)
    assert all_x.shape[0] == n_samples

    # save data
    dict_to_save = {
        "theta": all_theta,
        "x": all_x,
        "prior_params": prior_params,
        "prior_params_darcy": prior_params_darcy_gp,
        "prior_mode": prior_mode,
    }

    print(data_dir)
    save_path = Path(data_dir) / cfg.data_config.path / cfg.data_config.data_file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(dict_to_save, f)
    print("data saved.")


if __name__ == "__main__":
    print("started sampling....")
    sample_darcy()

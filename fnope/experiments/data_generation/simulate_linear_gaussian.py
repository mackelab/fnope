import torch
from omegaconf import DictConfig
from fnope.simulators.gp_priors import get_gaussian_process_prior_1d
from fnope.simulators.simulator import linear_gaussian
from fnope.utils.misc import get_data_dir,set_seed
import pickle
import hydra
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base="1.3", config_path="../../config", config_name="data_generation")
def sample_linear_gaussian(cfg: DictConfig):

    data_dir = get_data_dir()
    seed = cfg.data_config.seed
    set_seed(seed)

    # hyperparameters: might want to move to config file
    # prior
    prior_lengthscale = cfg.data_config.prior_lengthscale # 0.05 for original experiments
    prior_sigma = cfg.data_config.prior_sigma # 1.0 for original experiments

    # time setting
    seq_len = 1000  # number of time steps
    T = 1.0  # time horizon

    #simulator settings
    likelihood_shift = torch.zeros(seq_len).to(device)
    likelihood_cov = torch.eye(seq_len).to(device) * 0.1


    N = cfg.data_config.N

    # set seed
    torch.manual_seed(seed)

    # define prior
    ts = torch.linspace(0, T, seq_len)
    gp_prior = get_gaussian_process_prior_1d(num_points=seq_len,
                                             domain_length=T,
                                             mean=0.0, 
                                             lengthscale=prior_lengthscale, 
                                             sigma=prior_sigma, 
                                             device=device)


    # sample from prior
    theta = gp_prior.sample(torch.Size([N])).to(device)

    # run simulator
    x = linear_gaussian(theta,likelihood_shift=likelihood_shift, 
                        likelihood_cov=likelihood_cov)


    # save data
    dict_to_save = {
            "theta": theta,
            "x": x,
            "simulation_grid": ts,
        }
    print(data_dir)
    save_path = Path(data_dir) / cfg.data_config.path / cfg.data_config.data_file
    print(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
            pickle.dump(dict_to_save, f)
    print("data saved.")


if __name__ == "__main__":
    print("started sampling....")
    sample_linear_gaussian()

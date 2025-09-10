import torch
import numpy as np
import hydra
import pickle
import torch
from pathlib import Path
from fnope.utils.misc import get_data_dir,get_output_dir,set_seed


from fnope.flow_matching.fnope_1D import FNOPE_1D
from fnope.flow_matching.training import train_fnope
from fnope.simulators.gp_priors import get_gaussian_process_prior_1d
from fnope.experiments.evaluation_utils import run_sbc_save_results, run_tarp_save_results,run_swd_save_results, run_predictive_checks_save_results, FNOPosterior, GTPosterior
from fnope.simulators.simulator import linear_gaussian
from fnope.utils.misc import read_pickle

from omegaconf import DictConfig, OmegaConf
from time import time,time_ns
import pandas as pd



@hydra.main(version_base="1.3", config_path="../../config", config_name="linear_gaussian_fnope_config")
def run_fnope(cfg: DictConfig):
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir()
    out_dir = get_output_dir()

    folder_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    if (folder_path / "random_seed.csv").exists():
        random_random_seed = np.loadtxt(folder_path / "random_seed.csv", delimiter=",")
        random_random_seed = int(random_random_seed)
    else:
        random_random_seed = np.random.randint(2**16)
        np.savetxt(folder_path / "random_seed.csv", np.array([random_random_seed]), delimiter=",")

    set_seed(random_random_seed)
    if (folder_path / "config.yaml").exists():
        with open(folder_path / "config.yaml", "r") as f:
            cfg = OmegaConf.load(f)
    else:
        with open(folder_path / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
    # load data
    with open(
        Path(data_dir/cfg.data_config.data_path/cfg.data_config.train_data), "rb"
    ) as f:
        data = pickle.load(f)


    n_train = int(cfg.data_config.n_train)
    if n_train < 0:
        n_train = len(data["theta"])
    random_perm = torch.randperm(len(data["theta"]))
    theta = data["theta"][random_perm][:n_train]
    x = data["x"][random_perm][:n_train]
    sim_grid = data["simulation_grid"]

    # load test data
    with open(
        Path(data_dir/cfg.data_config.data_path/cfg.data_config.test_data), "rb"
    ) as f:
        data_test = pickle.load(f)
    theta_test = data_test["theta"]
    x_test = data_test["x"]

    modes = cfg.model_config.modes



    seq_len = sim_grid.numel()

    #Make sure simulation grid normalized to [0,1]
    simulation_positions = (sim_grid-sim_grid[0]) / (sim_grid[-1] - sim_grid[0])

    #Make sure data has channel dimension
    if theta.ndim == 2:
        theta = theta.unsqueeze(1)
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if theta_test.ndim == 2:
        theta_test = theta_test.unsqueeze(1)
    if x_test.ndim == 2:
        x_test = x_test.unsqueeze(1)

    theta = theta.to(device)
    x = x.to(device)
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)
    simulation_positions = simulation_positions.to(device)



    model_path = Path(folder_path) / "model.pkl"
    if model_path.exists():
        model = read_pickle(model_path)
        print("Model loaded from file")
    else:

        model = FNOPE_1D(
            x=theta,
            ctx=x,
            simulation_grid=simulation_positions,
            x_finite=None,
            modes=modes,
            conv_channels=cfg.model_config.conv_channels,
            ctx_embedding_channels=cfg.model_config.ctx_embedding_channels,
            time_embedding_channels=cfg.model_config.time_embedding_channels,
            position_embedding_channels=cfg.model_config.position_embedding_channels,
            training_point_noise=dict(cfg.model_config.training_point_noise),
            num_layers=cfg.model_config.num_layers,
            always_equispaced=cfg.model_config.always_equispaced,  # this is the default value, writing explicitly for clarity
            always_match_x_theta=False,  # this is the default value, writing explicitly for clarity
        ).to(device)
        nn_num_params = sum(p.numel() for p in model.parameters())
        np.savetxt(
            Path(folder_path) / "nn_num_params.csv",
            np.array([nn_num_params]),
            delimiter=","
        )
        time_start = time_ns()
        num_epochs, loss,val_loss = train_fnope(
            model=model,
            cfg=cfg,
            theta=theta,
            x=x,
            simulation_positions=simulation_positions,
            x_finite=None,
            save_path=folder_path,
            device=device,
        )
        time_end = time_ns()
        np.savetxt(
            Path(folder_path) / "training_time.csv",
            np.array([num_epochs,(time_end - time_start) / 1e9]),
            delimiter=","
        )


        # save model
        #save with torch.save instead?
        with open(
            model_path,
            "wb",
        ) as f:
            pickle.dump(model, f)


    model.eval()
    fno_posterior = FNOPosterior(model=model,theta_shape = (1,seq_len),x_shape=(1,seq_len),point_positions=simulation_positions)
    #Set random seed again for evaluation in case we are loading a pretrained model
    set_seed(random_random_seed)

    likelihood_shift = torch.zeros(seq_len).to(device)
    likelihood_cov = torch.eye(seq_len).to(device) * 0.1
    prior_lengthscale = 0.05
    prior_sigma = 1.0



    # define prior
    T = 1.0
    gp_prior = get_gaussian_process_prior_1d(num_points=seq_len,
                                             domain_length=T,
                                             mean=0.0,
                                             lengthscale=prior_lengthscale,
                                             sigma=prior_sigma, 
                                             device=device)
    gt_posterior = GTPosterior(likelihood_shift=likelihood_shift,
                               likelihood_cov=likelihood_cov,
                               gp_prior=gp_prior,
                               device=device)


    # run SBC
    print("Running SBC...")
    n_sbc_marginals = cfg.evaluation_config.n_sbc_marginals
    downsampling_scale = seq_len // n_sbc_marginals


    n_sbc = cfg.evaluation_config.n_sbc
    run_sbc_save_results(
        theta_test[:n_sbc],
        x_test[:n_sbc],
        fno_posterior,
        num_posterior_samples=cfg.evaluation_config.num_posterior_samples,
        downsampling_scale = downsampling_scale,
        path_to_save=str(folder_path),
    )
    print("SBC done")

    # run TARP
    print("Running TARP...")
    n_tarp = cfg.evaluation_config.n_tarp
    run_tarp_save_results(
        theta_test[:n_tarp],
        x_test[:n_tarp],
        fno_posterior,
        reference_points=None,
        num_posterior_samples=cfg.evaluation_config.num_posterior_samples,
        path_to_save=str(folder_path),
    )
    print("TARP done")


    # run SWD
    print("Running SWD...")
    n_swd = cfg.evaluation_config.n_swd
    run_swd_save_results(
        x_test[:n_swd],
        fno_posterior,
        gt_posterior,
        num_posterior_samples=cfg.evaluation_config.num_posterior_samples,
        path_to_save=str(folder_path),
        device=device,
    )
    print("SWD done")


    # run posterior predictive checks
    print("Running posterior predictive checks...")

    n_predictive = cfg.evaluation_config.n_predictive_samples
    num_posterior_samples = cfg.evaluation_config.n_predictive_posterior_samples

    posterior_samples = torch.zeros(
        num_posterior_samples, n_predictive, theta.shape[-1]
    ).to(device)

    posterior_predictive_samples = torch.zeros(
        num_posterior_samples, n_predictive, x.shape[-1]
    ).to(device)

    df = pd.DataFrame(columns=["num_samples", "time_ns"])
    num_samples_list = []
    time_ns_list = []

    for i in range(n_predictive):
        print(f"Sample {i}")
        # sample from posterior
        time_start = time_ns()
        posterior_samples[:, i, :] = fno_posterior.sample(
            num_posterior_samples, x_test[i].view(1, seq_len)
        ).squeeze()
        time_end = time_ns()
        num_samples_list.append(num_posterior_samples)
        time_ns_list.append(time_end - time_start)
        posterior_predictive_samples[:, i, :] = linear_gaussian(
            posterior_samples[:, i, :],
            likelihood_shift=likelihood_shift,
            likelihood_cov=likelihood_cov,
        ).squeeze()
    df["num_samples"] = np.array(num_samples_list)
    df["time_ns"] = np.array(time_ns_list)
    df.to_csv(folder_path / "sampling_times.csv", index=False)
    

    run_predictive_checks_save_results(
        x_test[:n_predictive],
        posterior_predictive_samples,
        path_to_save=str(folder_path),
    )
    print("Posterior predictive checks done")
    print("Completed all evaluations successfully!")

if __name__ == "__main__":
    run_fnope()

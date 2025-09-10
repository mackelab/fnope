import torch
import numpy as np
import hydra
import pickle
from omegaconf import DictConfig, OmegaConf
from sbi.neural_nets import posterior_nn,flowmatching_nn
from sbi.inference import SNPE,FMPE
from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding
from fnope.experiments.baseline_sbi_utils import perform_rfft_and_process
from fnope.experiments.evaluation_utils import run_sbc_save_results, run_tarp_save_results,run_swd_save_results, run_predictive_checks_save_results, SBIPosterior, GTPosterior
from fnope.simulators.gp_priors import get_gaussian_process_prior_1d
from fnope.simulators.simulator import linear_gaussian
from fnope.nets.standardizing_net import FiniteStandardizing,FilterStandardizing, IdentityStandardizing

from pathlib import Path
from fnope.utils.misc import get_data_dir,get_output_dir,set_seed
from time import time,time_ns
import pandas as pd


@hydra.main(version_base="1.3", config_path="../../config", config_name="linear_gaussian_baseline_sbi_config")
def run_baseline(cfg: DictConfig):
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


    n_train = cfg.data_config.n_train
    if n_train < 0:
        n_train = len(data["theta"])
    random_perm = torch.randperm(len(data["theta"]))
    theta_raw = data["theta"][random_perm][:n_train]
    x_raw = data["x"][random_perm][:n_train]

    #ensure data has no channel dimension
    if len(theta_raw.shape) == 3:
        theta_raw = theta_raw.squeeze(1)
    if len(x_raw.shape) == 3:
        x_raw = x_raw.squeeze(1)
    
    #sim_grid = data["simulation_grid"][:n_train]
    sim_grid = data["simulation_grid"]
    seq_len = sim_grid.numel()

    # load test data
    with open(
        Path(data_dir/cfg.data_config.data_path/cfg.data_config.test_data), "rb"
    ) as f:
        data_test = pickle.load(f)
    theta_raw_test = data_test["theta"]
    x_raw_test = data_test["x"]
    # #ensure data has no channel dimension
    # if len(theta_raw_test.shape) == 3:
    #     theta_raw_test = theta_raw_test.squeeze(1)
    # if len(x_raw_test.shape) == 3:
    #     x_raw_test = x_raw_test.squeeze(1)

    theta_raw = theta_raw.to(device)
    x_raw = x_raw.to(device)
    theta_raw_test = theta_raw_test.to(device)
    x_raw_test = x_raw_test.to(device)

    # data mode:
    if cfg.model_config.data_representation.lower() == "fourier":
        # currently only supports 1d data
        # and only theta is transformed to fourier space
        # fft and extend to real space with double the size
        theta = perform_rfft_and_process(
            theta_raw,
            cfg.model_config.n_fft_modes,
            pad_width=cfg.model_config.pad_width,
        )
        theta_test = perform_rfft_and_process(
            theta_raw_test,
            cfg.model_config.n_fft_modes,
            pad_width=cfg.model_config.pad_width,
        )

        x = x_raw
        x_test = x_raw_test
        z_scoring = "independent"
    else:
        x = x_raw
        theta = theta_raw
        x_test = x_raw_test
        theta_test = theta_raw_test
        z_scoring="structured"

    # set up embedding and sbi
    if cfg.model_config.embedding == "cnn":
        embedding_net = CNNEmbedding(
            input_shape=(x.shape[-1],),
            num_conv_layers=cfg.model_config.num_conv_layers,
            out_channels_per_layer = cfg.model_config.out_channels_per_layer,
            num_linear_layers=cfg.model_config.num_linear_layers,
            num_linear_units=cfg.model_config.num_linear_units,
            output_dim=cfg.model_config.embedding_dim,
            kernel_size=cfg.model_config.kernel_size,
            pool_kernel_size=cfg.model_config.pool_kernel_size,
        )

    elif cfg.model_config.embedding == "fc":
        embedding_net = FCEmbedding(
            input_dim=x.shape[-1],
            output_dim=cfg.model_config.embedding_dim,
            num_layers=cfg.model_config.embedding_num_layers,
            num_hiddens=cfg.model_config.embedding_hidden_dim,
        )

    elif cfg.model_config.embedding.lower() == "none":
        embedding_net = torch.nn.Identity()
    else:
        raise ValueError("embedding net not recognized")
    

    posterior_path = Path(folder_path) / "posterior.pkl"
    standardizing_net_path = Path(folder_path) / "standardizing_net.pkl"


    if standardizing_net_path.exists():
        with open(standardizing_net_path, "rb") as f:
            standardizing_net = pickle.load(f)
        print("Standardizing net loaded from file")

    if posterior_path.exists():
        with open(posterior_path, "rb") as f:
            posterior = pickle.load(f)
        print("Posterior loaded from file")

    
    else:

        if cfg.model_config.method.lower() == "npe":
            
            standardizing_net = IdentityStandardizing() #Standarizing done inside the sbi model

            density_estimator = posterior_nn(
                model="nsf",
                embedding_net=embedding_net,
                device=device,
                z_score_theta=z_scoring,
                z_score_x="structured",
            )

            inference = SNPE(prior=None, density_estimator=density_estimator, device=device)
            inference = inference.append_simulations(theta, x)
            print("starting training...")
            # train the density estimator and build the posterior
            time_start = time_ns()
            density_estimator = inference.train(
                max_num_epochs=1_000,
                training_batch_size=cfg.model_config.batch_size,
                show_train_summary=True,
            )
            time_end = time_ns()
            num_epochs = inference.summary["epochs_trained"][-1]
            #Save total time and number of epochs
            np.savetxt(
                Path(folder_path) / "training_time.csv",
                np.array([num_epochs,(time_end - time_start) / 1e9]),
                delimiter=","
            )
            nn_num_params = sum(p.numel() for p in inference._neural_net.parameters())
            np.savetxt(
                Path(folder_path) / "nn_num_params.csv",
                np.array([nn_num_params]),
                delimiter=","
            )


            posterior = inference.build_posterior(density_estimator)

        elif cfg.model_config.method.lower() =="fmpe":

            #Need to do z-scoring outside of sbi as the sbi implementation z-scoring for fmpe is currently broken
            # See https://github.com/sbi-dev/sbi/pull/1544
            if cfg.model_config.data_representation.lower() == "fourier":
                standardizing_net = FiniteStandardizing(theta)
                theta = standardizing_net.standardize(theta)

            elif cfg.model_config.data_representation.lower() == "raw":
                theta = theta.unsqueeze(1)
                standardizing_net = FilterStandardizing(theta,point_positions=None,num_channels=1,modes = cfg.model_config.n_fft_modes,ncutoff=False)
                theta = standardizing_net.standardize(theta,point_positions=None).squeeze(1)

            

            net_builder = flowmatching_nn(
                model=cfg.model_config.model_type,
                num_blocks=cfg.model_config.num_blocks,
                num_layers=cfg.model_config.num_layers,
                hidden_features=cfg.model_config.hidden_features,
                num_frequencies=cfg.model_config.num_freqs,
                embedding_net=embedding_net,
                z_scoring_theta="none", # See https://github.com/sbi-dev/sbi/pull/1544
                z_scoring_x="structured",

                # z_score_theta=z_scoring,
                # z_score_x=z_scoring,
            )

            inference = FMPE(prior=None, density_estimator=net_builder, device=device)
            time_start = time_ns()
            inference.append_simulations(theta, x).train(training_batch_size=cfg.model_config.batch_size, learning_rate=cfg.model_config.learning_rate, max_num_epochs=1_000)
            time_end = time_ns()
            num_epochs = inference.summary["epochs_trained"][-1]

            np.savetxt(
                Path(folder_path) / "training_time.csv",
                np.array([num_epochs,(time_end - time_start) / 1e9]),
                delimiter=","
            )
            nn_num_params = sum(p.numel() for p in inference._neural_net.parameters())
            np.savetxt(
                Path(folder_path) / "nn_num_params.csv",
                np.array([nn_num_params]),
                delimiter=","
            )
            posterior = inference.build_posterior()




        # save posterior and information
        with open(
            posterior_path,
            "wb",
        ) as f:
            pickle.dump(posterior, f)



        with open(
            Path(folder_path) / "training_summary.pkl",
            "wb",
        ) as f:
            pickle.dump(
                inference.summary,
                f,
            )

        with open(
            Path(folder_path) / "standardizing_net.pkl",
            "wb",
        ) as f:
            pickle.dump(
                standardizing_net,
                f,
            )
    

    wrapped_posterior_path = Path(folder_path) / "wrapped_posterior.pkl"
    if wrapped_posterior_path.exists():
        with open(wrapped_posterior_path, "rb") as f:
            sbi_posterior = pickle.load(f)
        print("Wrapped Posterior loaded from file")
    else:
        sbi_posterior = SBIPosterior(posterior=posterior,
                                data_representation=cfg.model_config.data_representation,
                                theta_shape=(1,seq_len),
                                x_shape=(1,seq_len),
                                theta_standardizing_net=standardizing_net,
                                theta_pad_width=cfg.model_config.pad_width
                                )
        with open(
            wrapped_posterior_path,
            "wb",
        ) as f:
            pickle.dump(sbi_posterior, f)

    sbi_posterior.to(device)

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
    n_sbc = cfg.evaluation_config.n_sbc
    n_sbc_marginals = cfg.evaluation_config.n_sbc_marginals
    downsampling_scale = seq_len // n_sbc_marginals

    run_sbc_save_results(
        theta_raw_test[:n_sbc],
        x_test[:n_sbc],
        sbi_posterior,
        downsampling_scale=downsampling_scale,
        num_posterior_samples=cfg.evaluation_config.num_posterior_samples,
        path_to_save=str(folder_path),
    )
    print("SBC done")
    print("Running TARP...")
    # run tarp
    n_tarp = cfg.evaluation_config.n_tarp
    run_tarp_save_results(
        theta_raw_test[:n_tarp],
        x_test[:n_tarp],
        sbi_posterior,
        reference_points=None,
        num_posterior_samples=cfg.evaluation_config.num_posterior_samples,
        path_to_save=str(folder_path),
    )
    print("TARP done")
    print("Running SWD...")
    
    # run SWD
    n_swd = cfg.evaluation_config.n_swd
    run_swd_save_results(
        x_test[:n_swd],
        sbi_posterior,
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
        num_posterior_samples, n_predictive, seq_len
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
        posterior_samples[:, i, :] = sbi_posterior.sample(
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
    run_baseline()

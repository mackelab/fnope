import torch
import numpy as np
import hydra
import pickle
from omegaconf import DictConfig, OmegaConf
from sbi.neural_nets import posterior_nn, flowmatching_nn
from sbi.inference import SNPE, FMPE
from sbi.neural_nets.embedding_nets import FCEmbedding, CNNEmbedding
from fnope.experiments.baseline_sbi_utils import perform_rfft_and_process
from fnope.simulators.darcy_utils import DarcyGPPrior
from fnope.experiments.evaluation_utils import (
    run_sbc_save_results,
    run_tarp_save_results,
    run_predictive_checks_save_results,
    SBIPosterior,
)
from fnope.simulators.darcy import Darcy2D
from fnope.nets.standardizing_net import (
    FiniteStandardizing,
    FilterStandardizing2d,
    IdentityStandardizing,
)

from pathlib import Path
from fnope.utils.misc import get_data_dir, get_output_dir, set_seed
from time import time,time_ns
import pandas as pd

@hydra.main(
    version_base="1.3", config_path="../../config", config_name="darcy_baseline_sbi_config"
)
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
        Path(data_dir / cfg.data_config.data_path / cfg.data_config.train_data), "rb"
    ) as f:
        data = pickle.load(f)

    n_train = cfg.data_config.n_train
    if n_train < 0:
        n_train = len(data["theta"])
    random_perm = torch.randperm(len(data["theta"]))
    theta_raw = data["theta"][random_perm][:n_train]
    x_raw = data["x"][random_perm][:n_train]

    # ensure data has no channel dimension
    if len(theta_raw.shape) == 3:
        theta_raw = theta_raw.squeeze(1)
    if len(x_raw.shape) == 3:
        x_raw = x_raw.squeeze(1)

    # sim_grid = data["simulation_grid"][:n_train]
    # seq_len = sim_grid.numel()
    theta_space_shape = theta_raw.shape[1:]
    x_space_shape = x_raw.shape[1:]
    print("theta space shape", theta_space_shape)
    print("x space shape", x_space_shape)

    # load test data
    with open(
        Path(data_dir / cfg.data_config.data_path / cfg.data_config.test_data), "rb"
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
        theta, (H_pad, W_pad) = perform_rfft_and_process(
            theta_raw,
            cfg.model_config.n_fft_modes,
            pad_width=cfg.model_config.pad_width,
        )
        
        theta_test, (H_pad, W_pad) = perform_rfft_and_process(
            theta_raw_test,
            cfg.model_config.n_fft_modes,
            pad_width=cfg.model_config.pad_width,
        )

        x = x_raw.to(device)
        x_test = x_raw_test
        z_scoring = "independent"
    else:
        x = x_raw.to(device)
        theta = theta_raw
        x_test = x_raw_test
        theta_test = theta_raw_test
        z_scoring = "structured"
        (H_pad, W_pad) = (0, 0)

    # set up embedding and sbi
    if cfg.model_config.embedding == "cnn":
        embedding_net = CNNEmbedding(
            input_shape=tuple(x_space_shape),
            num_conv_layers=cfg.model_config.num_conv_layers,
            out_channels_per_layer = cfg.model_config.out_channels_per_layer,
            num_linear_layers=cfg.model_config.num_linear_layers,
            num_linear_units=cfg.model_config.num_linear_units,
            output_dim=cfg.model_config.embedding_dim,
            kernel_size=5,
            pool_kernel_size=2,
            # out_channels_per_layer=[6,12], # [6,12] is default for 2 conv layers. adapt accordingly for more layers
        )

    elif cfg.model_config.embedding == "fc":
        raise ValueError("FC embedding not implemented yet for 2d data")
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

    print("x shape", x.shape)
    print("theta shape", theta.shape)
    print("x_test shape", x_test.shape)
    print("theta_test shape", theta_test.shape)

    posterior_path = Path(folder_path) / "posterior.pkl"
    standardizing_net_path = Path(folder_path) / "standardizing_net.pkl"

    if posterior_path.exists():
        with open(posterior_path, "rb") as f:
            posterior = pickle.load(f)
        print("Posterior loaded from file")
        with open(standardizing_net_path, "rb") as f:
            standardizing_net = pickle.load(f)
        print("Standardizing net loaded from file")
    else:

        if cfg.model_config.method.lower() == "npe":

            standardizing_net = (
                IdentityStandardizing()
            )  # Standarizing done inside the sbi model

            density_estimator = posterior_nn(
                model="nsf",
                embedding_net=embedding_net,
                device=device,
                z_score_theta=z_scoring,
                z_score_x="structured",
            )

            inference = SNPE(
                prior=None, density_estimator=density_estimator, device=device
            )
            theta = theta.to(device)
            x = x.to(device)
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

        elif cfg.model_config.method.lower() == "fmpe":

            # Need to do z-scoring outside of sbi as the sbi implementation z-scoring for fmpe is currently broken
            # See https://github.com/sbi-dev/sbi/pull/1544
            if cfg.model_config.data_representation.lower() == "fourier":
                standardizing_net = FiniteStandardizing(theta)
                theta = standardizing_net.standardize(theta).to(device)

            elif cfg.model_config.data_representation.lower() == "raw":
                theta = theta.unsqueeze(1).to(device)
                standardizing_net = FilterStandardizing2d(
                    theta,
                    point_positions=None,
                    num_channels=1,
                    modes=cfg.model_config.n_fft_modes,
                    ncutoff=False,
                )
                theta = standardizing_net.standardize(
                    theta, point_positions=None
                ).squeeze(1)

                # # theta needs to be 1D for sbi input
                theta = theta.reshape(theta.shape[0], -1)

            net_builder = flowmatching_nn(
                model=cfg.model_config.model_type,
                num_blocks=cfg.model_config.num_blocks,
                num_layers=cfg.model_config.num_layers,
                hidden_features=cfg.model_config.hidden_features,
                num_frequencies=cfg.model_config.num_freqs,
                embedding_net=embedding_net,
                z_scoring_theta="none",  # See https://github.com/sbi-dev/sbi/pull/1544
                z_scoring_x="structured",
                # z_score_theta=z_scoring,
                # z_score_x=z_scoring,
            )
            theta = theta.to(device)
            x = x.to(device)
            print(device)
            inference = FMPE(prior=None, density_estimator=net_builder, device=device)
            time_start = time_ns()

            inference.append_simulations(theta, x).train(
                training_batch_size=cfg.model_config.batch_size,
                learning_rate=cfg.model_config.learning_rate,
                max_num_epochs=1_000,
            )
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
                                theta_shape=(1,)+tuple(theta_space_shape),
                                x_shape=(1,)+tuple(x_space_shape),
                                theta_standardizing_net=standardizing_net,
                                theta_pad_width=(H_pad, W_pad), 
                                x_dims=2,
                                theta_dims=2
                                )
        with open(
            wrapped_posterior_path,
            "wb",
        ) as f:
            pickle.dump(sbi_posterior, f)
        print("Wrapped Posterior saved to file")

    sbi_posterior.to(device)
    #Set random seed again for evaluation in case we are loading a pretrained model
    set_seed(random_random_seed)
    post_samples = sbi_posterior.sample(10, x_test[0])

    print("Posterior samples shape", post_samples.shape)

    # # run SBC

    print("Running SBC...")
    n_sbc = cfg.evaluation_config.n_sbc
    n_sbc_marginals = cfg.evaluation_config.n_sbc_marginals
    downsampling_scale = (theta_space_shape[0]*theta_space_shape[1]) // n_sbc_marginals


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

    # run posterior predictive checks
    print("Running posterior predictive checks...")


    # get hyperparameters
    resolution = x_space_shape[-1] - 1



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up simulator
    darcy = Darcy2D(
        batch_size=cfg.evaluation_config.n_predictive_posterior_samples,
        resolution=resolution,
        snr=cfg.data_config.snr,
        #prior config does not matter as we are not using prior
    )

    n_predictive = cfg.evaluation_config.n_predictive_samples
    num_posterior_samples = cfg.evaluation_config.n_predictive_posterior_samples

    posterior_samples = torch.zeros(
        num_posterior_samples, n_predictive, *theta_space_shape
    ).to(device)

    posterior_predictive_samples = torch.zeros(
        num_posterior_samples, n_predictive, *x_space_shape
    ).to(device)
    df = pd.DataFrame(columns=["num_samples", "time_ns"])
    num_samples_list = []
    time_ns_list = []
    for i in range(n_predictive):
        print(f"Sample {i}")
        # sample from posterior
        time_start = time_ns()

        posterior_samples[:, i, :] = sbi_posterior.sample(
            num_posterior_samples, x_test[i]
        ).squeeze()
        time_end = time_ns()
        num_samples_list.append(num_posterior_samples)
        time_ns_list.append(time_end - time_start)
        t_temp,t_res_temp,x_temp = darcy.simulate_darcy(
            posterior_samples[:, i, :, :],
        )

        posterior_predictive_samples[:, i, :, :] = x_temp
    
    df["num_samples"] = np.array(num_samples_list)
    df["time_ns"] = np.array(time_ns_list)
    df.to_csv(folder_path / "sampling_times.csv", index=False)

    run_predictive_checks_save_results(
        x_test[:n_predictive],
        posterior_predictive_samples,
        path_to_save=str(folder_path),
    )
    print("Posterior predictive checks done")
    print("Saving predictive summary...")


    predictive_summary = {
        "theta_test": theta_test[:n_predictive].detach().cpu().numpy(),
        "x_test": x_test[:n_predictive].detach().cpu().numpy(),
        "posterior_samples": posterior_samples.detach().cpu().numpy(),
        "posterior_predictive_samples": posterior_predictive_samples.detach().cpu().numpy(),
    }
    with open(
        Path(folder_path / "predictive_summary.pkl"), "wb"
    ) as f:
        pickle.dump(predictive_summary, f)

    #Calculate prior log probs of posterior samples
    darcy_prior = DarcyGPPrior(
        n = posterior_samples.shape[-1],
        alpha=cfg.data_config.alpha,
        tau = cfg.data_config.tau,
        scale = cfg.data_config.prior_scale,
        device = device
    )


    print("Calculating prior log probs...")
    prior_log_probs = torch.zeros(num_posterior_samples,n_predictive).to(device)
    for i in range(n_predictive):
        prior_log_probs[:,i] = darcy_prior.log_prob(
            posterior_samples[:,i]
        )
    with open(
        Path(folder_path / "prior_log_probs.pkl"), "wb"
    ) as f:
        pickle.dump({"prior_log_probs":prior_log_probs.cpu()}, f)
    
    print("Prior log probs done")
    print(prior_log_probs)
    if cfg.model_config.name.lower() == "raw_fmpe":
        print("Calculating Posterior log probs...")
        with torch.no_grad():
            sbi_logprobs = sbi_posterior.log_prob(theta_test[:n_predictive],x = x_test[:n_predictive])
            correction_factor = theta_test[0].numel()*np.log(sbi_posterior.theta_standardizing_net.channelwise_power.item())
            sbi_logprobs = sbi_logprobs - correction_factor
            sbi_logprobs_per_pixel = sbi_logprobs/theta_test[0].numel()
        with open(
            Path(folder_path / "posterior_log_probs.pkl"), "wb"
        ) as f:
            pickle.dump(
                {
                    "posterior_log_probs": sbi_logprobs.cpu(),
                    "posterior_log_probs_per_pixel": sbi_logprobs_per_pixel.cpu(),

                }
                , f)


        print("Posterior log probs done")
        print(sbi_logprobs_per_pixel)
    print("Completed all evaluations successfully!")


if __name__ == "__main__":
    run_baseline()

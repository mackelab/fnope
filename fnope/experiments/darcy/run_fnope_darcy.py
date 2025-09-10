import torch
import numpy as np
import hydra
import pickle
from omegaconf import DictConfig, OmegaConf
from fnope.experiments.evaluation_utils import run_sbc_save_results, run_tarp_save_results,run_predictive_checks_save_results, SBIPosterior, FNOPosterior
from fnope.flow_matching.training import train_fnope
from fnope.flow_matching.fnope_2D import FNOPE_2D
from pathlib import Path
from fnope.simulators.darcy import Darcy2D
from fnope.simulators.darcy_utils import DarcyGPPrior
from fnope.utils.misc import get_data_dir,get_output_dir,set_seed,read_pickle

import traceback
import sys
from time import time,time_ns
import pandas as pd


@hydra.main(version_base="1.3", config_path="../../config", config_name="darcy_fnope_config")
def run_fnope(cfg: DictConfig):
    try: 
        # get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_dir = get_data_dir()
        out_dir = get_output_dir()



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
        theta = data["theta"][random_perm][:n_train]
        x = data["x"][random_perm][:n_train]
        print("theta mean", theta.mean())
        print("theta std", theta.std())
        print("x mean", x.mean())
        print("x std", x.std())

        # load test data
        with open(
            Path(data_dir/cfg.data_config.data_path/cfg.data_config.test_data), "rb"
        ) as f:
            data_test = pickle.load(f)
        theta_test = data_test["theta"]
        x_test = data_test["x"]



        modes = cfg.model_config.modes


        #Make sure data has channel dimension
        if theta.ndim == 3:
            theta = theta.unsqueeze(1)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if theta_test.ndim == 3:
            theta_test = theta_test.unsqueeze(1)
        if x_test.ndim == 3:
            x_test = x_test.unsqueeze(1)


        theta_space_shape = theta.shape[1:]
        x_space_shape = x.shape[1:]
        n_theta = torch.prod(torch.tensor(theta_space_shape))
        n_x = torch.prod(torch.tensor(x_space_shape))

        theta = theta.to(device)
        x = x.to(device)
        theta_test = theta_test.to(device)
        x_test = x_test.to(device)



        theta_x_size,theta_y_size = theta.shape[-2:]
        
        simulation_positions = torch.meshgrid(torch.arange(0, theta_x_size)/theta_x_size, torch.arange(0, theta_y_size)/theta_y_size)
        simulation_positions = torch.stack(simulation_positions, dim=-1).to(device)
        print(f"simulation positions shape {simulation_positions.shape}")

        ctx_x_size,ctx_y_size = x.shape[-2:]
        ctx_simulation_positions = torch.meshgrid(torch.arange(0, ctx_x_size)/ctx_x_size, torch.arange(0, ctx_y_size)/ctx_y_size)
        ctx_simulation_positions = torch.stack(ctx_simulation_positions, dim=-1).to(device)
        print(f"ctx simulation positions shape {ctx_simulation_positions.shape}")

        simulation_grid = simulation_positions if not cfg.model_config.always_equispaced else None


        model_pkl_path = Path(folder_path) / "model.pkl"
        model_state_dict_path = Path(folder_path) / "model_state_dict.pth"
        if model_pkl_path.exists():
            model = read_pickle(model_pkl_path)
            print("Model loaded from file")
        else:
            if cfg.model_config.name.lower() == "fnope":
                model = FNOPE_2D(
                    x=theta,
                    ctx=x,
                    simulation_grid=simulation_grid,
                    x_finite=None,
                    modes=modes,
                    base_dist=cfg.model_config.base_dist if cfg.model_config.base_dist is not None else "gp",
                    base_dist_lengthscale_multiplier=cfg.model_config.base_dist_lengthscale_multiplier,
                    conv_channels=cfg.model_config.conv_channels,
                    ctx_embedding_channels=cfg.model_config.ctx_embedding_channels,
                    time_embedding_channels=cfg.model_config.time_embedding_channels,
                    position_embedding_channels=cfg.model_config.position_embedding_channels,
                    num_layers=cfg.model_config.num_layers,
                    training_point_noise=dict(cfg.model_config.training_point_noise),
                    always_equispaced=cfg.model_config.always_equispaced,  # this is the default value, writing explicitly for clarity
                    always_match_x_theta=False,  # this is the default value, writing explicitly for clarity
                ).to(device)

            else:
                raise NotImplementedError(f"Model {cfg.model_config.name} not implemented")
            
            if model_state_dict_path.exists():
                print("Loading model state dict from file")
                model.load_state_dict(torch.load(model_state_dict_path, map_location=device),strict=False)
                model.to(device)
            else:
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
                    ctx_simulation_positions=ctx_simulation_positions,
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
                # with open(
                #     model_path,
                #     "wb",
                # ) as f:
                #     pickle.dump(model, f)
                torch.save(model.state_dict(), model_state_dict_path)
                print("Model state dict saved to file")

        
        model.eval()
        fno_posterior = FNOPosterior(model=model,
                                    theta_shape = theta_space_shape,
                                    x_shape=x_space_shape,
                                    point_positions=simulation_positions,
                                    ctx_point_positions=ctx_simulation_positions,
                                    ndims=2,
                                    sampling_batch_size=cfg.evaluation_config.sampling_batch_size,
                                    )

        #Set random seed again for evaluation in case we are loading a pretrained model
        set_seed(random_random_seed)




        # run SBC

        print("Running SBC...")
        n_sbc = cfg.evaluation_config.n_sbc
        n_sbc_marginals = cfg.evaluation_config.n_sbc_marginals
        downsampling_scale = n_theta // n_sbc_marginals
        run_sbc_save_results(
            theta_test[:n_sbc],
            x_test[:n_sbc],
            fno_posterior,
            downsampling_scale=downsampling_scale,
            num_posterior_samples=cfg.evaluation_config.num_posterior_samples,
            path_to_save=str(folder_path),
        )
        print("SBC done")
        print("Running TARP...")
        # run tarp
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
    

        # run posterior predictive checks
        print("Running posterior predictive checks...")


        # get hyperparameters
        resolution = x_space_shape[-1] -1 



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
            num_posterior_samples, n_predictive, *theta_space_shape[1:]
        ).to(device)

        posterior_predictive_samples = torch.zeros(
            num_posterior_samples, n_predictive, *x_space_shape[1:]
        ).to(device)
        df = pd.DataFrame(columns=["num_samples", "time_ns"])
        num_samples_list = []
        time_ns_list = []
        for i in range(n_predictive):
            print(f"Sample {i}")
            # sample from posterior
            time_start = time_ns()
            posterior_samples[:, i, :] = fno_posterior.sample(
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
        print("saving...")


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
            n = theta_test.shape[-1],
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
            pickle.dump({"prior_log_probs" :prior_log_probs.cpu()}, f)
        
        print("Prior log probs done")
        print(prior_log_probs)
        print("Calculating Posterior log probs...")

        sampling_batch_size = cfg.evaluation_config.sampling_batch_size
        posterior_logprobs = torch.zeros(n_predictive).to(device)
        n_full_batches = n_predictive // sampling_batch_size
        last_batch_size = n_predictive % sampling_batch_size


        # Generate all batches and collect
        if n_full_batches >0:
            for i in range(n_full_batches):
                lps = model.unnormalized_log_prob(theta_test[i*sampling_batch_size:(i+1)*sampling_batch_size],
                                                ctx = x_test[i*sampling_batch_size:(i+1)*sampling_batch_size],
                                                x_finite = None,
                                                point_positions = simulation_positions,
                                                ctx_point_positions = ctx_simulation_positions)
                posterior_logprobs[i*sampling_batch_size:(i+1)*sampling_batch_size] = lps


        if last_batch_size > 0:
            lps = model.unnormalized_log_prob(theta_test[-last_batch_size:],
                                            ctx = x_test[-last_batch_size:],
                                            x_finite = None,
                                            point_positions = simulation_positions,
                                            ctx_point_positions = ctx_simulation_positions)
            posterior_logprobs[-last_batch_size:] = lps

        correction_factor = theta_test[0].numel()*np.log(model.x_standardizing_net.channelwise_power.item())
        posterior_logprobs = posterior_logprobs - correction_factor
        posterior_logprobs_per_pixel = posterior_logprobs/theta_test[0].numel()
        with open(
            Path(folder_path / "posterior_log_probs.pkl"), "wb"
        ) as f:
            pickle.dump(
                {
                    "posterior_log_probs": posterior_logprobs.cpu(),
                    "posterior_log_probs_per_pixel": posterior_logprobs_per_pixel.cpu(),

                }
                , f)

        print("Posterior log probs done")
        print(posterior_logprobs_per_pixel)

        print("Completed all evaluations successfully!")
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    run_fnope()


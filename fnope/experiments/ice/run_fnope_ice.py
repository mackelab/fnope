from time import time_ns
import torch
import numpy as np
import hydra
import pickle
from omegaconf import DictConfig, OmegaConf
from sbi.inference import SNPE
from sbi.utils import process_prior, BoxUniform
from sbi.neural_nets.estimators import NFlowsFlow
from fnope.experiments.evaluation_utils import run_sbc_save_results, run_tarp_save_results,run_swd_save_results, run_predictive_checks_save_results, SBIPosterior, FNOPosterior
from fnope.nets.standardizing_net import IdentityStandardizing
from fnope.flow_matching.training import train_fnope
from fnope.flow_matching.fnope_1D import FNOPE_1D


from pathlib import Path
import pandas as pd
from fnope.simulators.ice_simulator.modelling_utils import regrid
from fnope.simulators.ice_simulator.evaluate_posterior_predictive import simulate_layers, simulate_one_sample
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern
from fnope.utils.misc import get_data_dir,get_output_dir,set_seed,read_pickle


@hydra.main(version_base="1.3", config_path="../../config", config_name="ice_fnope_config")
def run_fnope(cfg: DictConfig):
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = get_data_dir()
    out_dir = get_output_dir()

    folder_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) #pyright: ignore
    print(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    if Path(folder_path / "random_seed.csv").exists():
        with open(folder_path / "random_seed.csv", "r") as f:
            random_random_seed = np.loadtxt(f, delimiter=",")
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



    # Mimicking how we load the data in the sbi_ice project

    setup_df = pd.read_csv(Path(data_dir/cfg.data_config.simulator.setup_file))
    x_coords = setup_df["x_coord"].to_numpy()
    surface = setup_df["surface"].to_numpy()
    print("x_coords",x_coords.shape)
    print("surface",surface.shape)
    sim_grid = np.linspace(x_coords[0],x_coords[-1],cfg.data_config.simulator.iso_hparams.nx_iso)

    layer_bounds = read_pickle(Path(data_dir/cfg.data_config.simulator.layer_bounds_file))
    masks = [sim_grid > bound for bound in layer_bounds]
    masks = np.array(masks)
    mask = masks[cfg.data_config.simulator.layer_idx]

    res = read_pickle(Path(data_dir/cfg.data_config.param_train_path))
    smb_unperturbed_all = res["smb_unperturbed_all"].to(torch.float32).to(device)
    smb_cnst_means_all = res["smb_cnst_means_all"].to(torch.float32).to(device)
    smb_sds_all = res["smb_sds_all"].to(torch.float32).to(device)
    smb_all = res["smb_all"].to(torch.float32).to(device)
    bmb_all = res["bmb_all"].to(torch.float32).to(device)


    res = read_pickle(Path(data_dir/cfg.data_config.observation_train_path))
    contour_arrays = res["contour_arrays"].to(torch.float32).to(device)
    norm_arrays = res["norm_arrays"].to(torch.float32).to(device)
    age_arrays = res["age_arrays"].to(torch.float32).to(device)
    

    layer_mask = torch.from_numpy(mask)
    layer_sparsity = cfg.data_config.simulator.grid.layer_sparsity
    smb_sparsity = cfg.data_config.simulator.grid.smb_sparsity

    layers = contour_arrays[cfg.data_config.simulator.layer_idx]
    layer_mask_slice = torch.zeros(layers.shape[-1], dtype=bool)#pyright: ignore
    layer_mask_slice[::layer_sparsity] = 1

    layer_all_mask = layer_mask * layer_mask_slice
    layers_all = layers[:,layer_all_mask].to(device).float()

    smb_mask_slice = torch.zeros(layers.shape[-1],dtype=bool)#pyright: ignore
    smb_mask_slice[::smb_sparsity] = 1
    #smb_mask = layer_mask*smb_mask_slice
    smb_mask = smb_mask_slice #infer all smb values
    smb_x = sim_grid[smb_mask]
    layer_x = sim_grid[layer_all_mask]
    seq_len = smb_x.shape[-1]


    n_test_holdout = cfg.data_config.n_test_holdout

    smb_test = smb_all[-n_test_holdout:,smb_mask]
    layers_test = layers_all[-n_test_holdout:,:]
    smb_train = smb_all[:-n_test_holdout,smb_mask]
    layers_train = layers_all[:-n_test_holdout,:]
    

    n_train = cfg.data_config.n_train
    if n_train < 0:
        n_train = smb_train.shape[0]

    random_perm = torch.randperm(smb_train.shape[0])
    smb_train = smb_train[random_perm,:]
    smb_train = smb_train[:n_train,:]
    layers_train = layers_train[random_perm,:]
    layers_train = layers_train[:n_train,:]



    #Load real layer
    layers_df = pd.read_csv(Path(data_dir/cfg.data_config.test_data))
    n_real_layers = len(layers_df.columns)-2
    print("number of real layers: "  , n_real_layers)
    real_layers = np.zeros(shape=(n_real_layers,cfg.data_config.simulator.iso_hparams.nx_iso))
    #Regrid the real layers to the simulation grid (e.g. real data is defined on a different grid)
    for i in range(n_real_layers):
        layer_depths = regrid(layers_df["x_coord"],layers_df["layer {}".format(i+1)],sim_grid,kind="linear")
        real_layers[i,:] = surface-layer_depths
    true_layer_unmasked = torch.tensor(real_layers[cfg.data_config.simulator.layer_idx]).float()
    true_layer = torch.tensor(real_layers[cfg.data_config.simulator.layer_idx][layer_all_mask]).float()

    #Define prior over SMB
    GP_mean_mu = torch.tensor([cfg.data_config.simulator.prior.GP_mean_mu],device=device)
    GP_mean_sd = torch.tensor([cfg.data_config.simulator.prior.GP_mean_sd],device=device)
    GP_var_min = torch.tensor([cfg.data_config.simulator.prior.GP_var_min],device=device)
    GP_var_max = torch.tensor([cfg.data_config.simulator.prior.GP_var_max],device=device)
    smb_prior_length_scale = cfg.data_config.simulator.prior.length_scale
    smb_prior_nu = cfg.data_config.simulator.prior.nu

    #Define GP kernel
    ker = Matern(length_scale=smb_prior_length_scale,nu=smb_prior_nu)
    gpr = GaussianProcessRegressor(kernel=ker)

    mvn_mean,mvn_cov = gpr.predict(smb_x.reshape(-1,1),return_cov=True)
    eps = 1e-6
    a = np.zeros(shape = mvn_cov.shape)
    np.fill_diagonal(a,eps)
    mvn_cov += a
    mvn_mean = torch.from_numpy(mvn_mean).to(device)
    mvn_cov = torch.from_numpy(mvn_cov).to(device)

    custom_prior = BoxUniform(low=-5.0*torch.ones(mvn_mean.size()), 
                                high=5.0*torch.ones(mvn_mean.size()),device=device.type)

    spatial_prior, *_ = process_prior(custom_prior,
                              custom_prior_wrapper_kwargs=dict(lower_bound=-5.0*torch.ones(mvn_mean.size()), 
                                                               upper_bound=5.0*torch.ones(mvn_mean.size())))




    theta = smb_train.to(device).float()
    x = layers_train.to(device)
    theta_test = smb_test.to(device)
    x_test = layers_test.to(device)

    modes = cfg.model_config.modes


    #Make sure simulation grid normalized to [0,1]
    simulation_positions = torch.from_numpy((smb_x-smb_x[0]) / (smb_x[-1] - smb_x[0])).to(device).to(torch.float32)
    ctx_simulation_positions = torch.from_numpy((layer_x-smb_x[0]) / (layer_x[-1] - smb_x[0])).to(device).to(torch.float32)
    # ctx_simulation_positions = torch.from_numpy((layer_x-layer_x[0]) / (layer_x[-1] - layer_x[0])).to(device).to(torch.float32)
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
        model.to(device)
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
            num_layers=cfg.model_config.num_layers,
            training_point_noise=dict(cfg.model_config.training_point_noise),
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
        #save with torch.save instead?
        with open(
            model_path,
            "wb",
        ) as f:
            pickle.dump(model, f)

    
    model.eval()
    model.to(device)
    fno_posterior = FNOPosterior(model=model,
                                 theta_shape = (1,seq_len),
                                 x_shape=(1,x.shape[-1]),
                                 point_positions=simulation_positions,
                                 ctx_point_positions=ctx_simulation_positions
                                )

    #Set random seed again for evaluation in case we are loading a pretrained model
    set_seed(random_random_seed)

    # load gt posterior from npe with ~200k sims from paper
    try:

        gt_inference_path = Path(data_dir/cfg.data_config.gt_inference_path / "inference.p")
        gt_inference = read_pickle(gt_inference_path)
    except:
        network_path = Path(data_dir/cfg.data_config.gt_inference_path / "neural_net.pth")
        neural_net = torch.load(network_path,map_location=device)
        nflows_flow = NFlowsFlow(net=neural_net,input_shape=(seq_len,),condition_shape=(x.shape[-1],)).to(device)
        gt_inference = SNPE(prior=spatial_prior,device=device)
        gt_inference.density_estimator = nflows_flow

        with open(Path(data_dir/cfg.data_config.gt_inference_path / "inference.p"), "wb") as f:
            pickle.dump(gt_inference, f)
    gt_posterior = gt_inference.build_posterior(density_estimator=gt_inference.density_estimator)
    gt_posterior = SBIPosterior(posterior=gt_posterior,
                                data_representation="raw",
                                theta_shape=(1,seq_len),
                                x_shape = (1,x.shape[-1]),
                                theta_standardizing_net=IdentityStandardizing(),
                                theta_pad_width=0
                                )


    # run SBC

    print("Running SBC...")
    n_sbc = cfg.evaluation_config.n_sbc
    n_sbc_marginals = cfg.evaluation_config.n_sbc_marginals
    downsampling_scale = seq_len // n_sbc_marginals
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
    if cfg.data_config.simulator.grid.smb_sparsity == 10:
        print("Running SWD...")
        # run SWD
        n_swd = cfg.evaluation_config.n_swd
        run_swd_save_results(
            x_test[:n_swd],
            fno_posterior,
            gt_posterior,
            num_posterior_samples=cfg.evaluation_config.num_posterior_samples,
            path_to_save=str(folder_path),
            device=device, #pyright: ignore
        )
        print("SWD done")
    else:
        print("Not running same settings as GT, skipping SWD")

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

    posterior_bmbs = torch.zeros(
        num_posterior_samples, n_predictive, seq_len
    ).to(device)

    df = pd.DataFrame(columns=["num_samples", "time_ns"])
    num_samples_list = []
    time_ns_list = []

    for i in range(n_predictive):
        print(f"Test Sample {i+1} of {n_predictive}")
        # sample from posterior
        time_start = time_ns()
        posterior_samples[:, i, :] = fno_posterior.sample(
            num_posterior_samples, x_test[i].view(1, x_test[i].shape[-1])
        ).squeeze()
        time_end = time_ns()
        num_samples_list.append(num_posterior_samples)
        time_ns_list.append(time_end - time_start)

        for j in range(num_posterior_samples):
            # sample from likelihood
            print("Simulating sample %d of %d" % (j+1,num_posterior_samples))
            best_contour, best_age, norm, bmb_regrid = simulate_one_sample(
                cfg=cfg,
                smb_x=smb_x,
                smb_sample=posterior_samples[j, i, :].cpu().numpy(),
                true_layer=true_layer_unmasked.cpu(),
                layer_mask=layer_mask.cpu(),
                selection_method="advanced_noise",
            )


            posterior_predictive_samples[j, i, :] = best_contour[layer_mask].to(torch.float32)
            posterior_bmbs[j, i, :] = torch.from_numpy(bmb_regrid.flatten()[smb_mask]).to(torch.float32)

    df["num_samples"] = np.array(num_samples_list)
    df["time_ns"] = np.array(time_ns_list)
    df.to_csv(folder_path / "sampling_times.csv", index=False)

    run_predictive_checks_save_results(
        x_test[:n_predictive],
        posterior_predictive_samples,
        path_to_save=str(folder_path),
    )

    with open(
        Path(folder_path) / "posterior_predictive_samples.pkl",
        "wb",
    ) as f:
        pickle.dump({"posterior_smbs":posterior_samples,
                     "posterior_bmbs":posterior_bmbs,
                     "posterior_predictives":posterior_predictive_samples}, f)
    # run posterior predictive checks
    print("Running posterior predictive checks on real layers...")

    num_posterior_samples = cfg.evaluation_config.n_predictive_posterior_samples_real_data

    posterior_samples = fno_posterior.sample(num_samples=num_posterior_samples,x=true_layer.unsqueeze(0).to(device))

    simulate_layers(
        cfg=cfg,
        smb_x = smb_x,
        smb_samples = posterior_samples.cpu().numpy(),
        true_layer = true_layer_unmasked.cpu(),
        layer_mask = layer_mask.cpu(),
        path_to_save=str(folder_path),
        selection_method = "advanced_noise",
    )
    print("Posterior predictive checks done")



    print("Completed all evaluations successfully!")
    


if __name__ == "__main__":
    run_fnope()

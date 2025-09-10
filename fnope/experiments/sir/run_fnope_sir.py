import torch
import numpy as np
import hydra
import pickle
import os
from omegaconf import DictConfig, OmegaConf
from fnope.experiments.evaluation_utils import check_tarp, run_predictive_checks_save_results
from sbi.diagnostics.sbc import _run_sbc, check_sbc
from sbi.diagnostics.tarp import _run_tarp, get_tarp_references
from sbi.utils.metrics import l2

from fnope.flow_matching.training import train_fnope
from fnope.flow_matching.fnope_1D import FNOPE_1D
from pathlib import Path
from fnope.simulators.simulator import SIR
from fnope.utils.misc import get_data_dir,get_output_dir,set_seed,read_pickle
from fnope.utils.sampling import rejection_sample
from time import time,time_ns
import pandas as pd

@hydra.main(version_base="1.3", config_path="../../config", config_name="sir_fnope_config")
def run_fnope(cfg: DictConfig):
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


    n_train = cfg.data_config.n_train
    if n_train < 0:
        n_train = len(data["theta"])

    train_data_name = f"train_data_{n_train//1000}k.npz"
    train_data_path = Path(data_dir / cfg.data_config.data_path / train_data_name)
    training_data = np.load(train_data_path)

    theta_times = training_data["metadata"][2:102]
    I_times = training_data["metadata"][102:202]
    R_times = training_data["metadata"][202:302]
    D_times = training_data["metadata"][302:]


    theta_fin = training_data["theta"][:, :2]
    theta_cont = training_data["theta"][:, 2:]
    I = training_data['x'][:, :100]
    R = training_data['x'][:, 100:200]
    D = training_data['x'][:, 200:300]
    train_x = np.stack([I, R, D], axis=1)

    sim_ts = torch.from_numpy(theta_times)
    beta = torch.from_numpy(theta_cont)



    T = sim_ts.max().item()


    theta = beta.unsqueeze(1).to(device)
    x = torch.from_numpy(train_x).clone().to(device)
    theta_finite = torch.from_numpy(theta_fin).to(device)
    simulation_positions = sim_ts.to(device)/T
    ctx_simulation_positions = simulation_positions

    modes = cfg.model_config.modes


    #Make sure parameters have channel dimension
    if theta.ndim == 2:
        theta = theta.unsqueeze(1)




    simulation_grid = simulation_positions if not cfg.model_config.always_equispaced else None


    model_path = Path(folder_path) / "model.pkl"
    if model_path.exists():
        model = read_pickle(model_path)
        print("Model loaded from file")
    else:

        model = FNOPE_1D(
            x=theta,
            ctx=x,
            simulation_grid=simulation_grid,
            x_finite=theta_finite,
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
            x_finite=theta_finite,
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
    #Set random seed again for evaluation in case we are loading a pretrained model
    set_seed(random_random_seed)


    # run SBC
    n_cond = cfg.evaluation_config.n_cond
    test_data_name = f"posterior{n_train//1000}k_samples_{n_cond}_time_points.npz"
    test_data = np.load(Path(data_dir / cfg.data_config.data_path / test_data_name))

    I = test_data['x_o'][:, :n_cond]
    R = test_data['x_o'][:, n_cond:2*n_cond]
    D = test_data['x_o'][:, 2*n_cond:3*n_cond]
    test_x = torch.from_numpy(np.stack([I, R, D], axis=1)).to(device)

    test_theta_finite = torch.from_numpy(test_data["theta_o"][:,:2]).to(device)
    test_theta_cont = torch.from_numpy(test_data["theta_o"][:,2:]).to(device)
    test_theta = torch.cat([test_theta_finite, test_theta_cont], dim=-1).to(device)
    test_times_theta = torch.from_numpy(test_data["meta_data"][:,2:2+n_cond]).to(device)
    test_times_x = torch.from_numpy(test_data["meta_data"][:,2+n_cond:2+2*n_cond]).to(device)
    simformer_initial_conditions = torch.from_numpy(test_data["posterior_initial_values"]).to(device)
    simformer_posterior_samples = torch.from_numpy(test_data["posterior_samples"]).to(device)





    print("Running SBC...")
    n_sbc = cfg.evaluation_config.n_sbc
    n_predictive = cfg.evaluation_config.n_predictive_samples
    num_posterior_samples = cfg.evaluation_config.n_predictive_posterior_samples

    posterior_samples = torch.zeros(
        num_posterior_samples, n_predictive, test_times_theta.shape[-1]+2
    ).to(device)

    simformer_posterior_samples = simformer_posterior_samples.permute(1,0,2)[:num_posterior_samples,:n_predictive]
    simformer_cont_samples = simformer_posterior_samples[...,2:]
    simformer_fin_samples = simformer_posterior_samples[...,:2]
    simformer_cont_samples = torch.clamp(simformer_cont_samples, min=0.0, max=1.0)
    simformer_fin_samples = torch.clamp(simformer_fin_samples, min=0.0, max=0.5)
    simformer_posterior_samples = torch.cat((simformer_fin_samples, simformer_cont_samples), dim=-1)

    posterior_predictive_samples = torch.zeros(
        num_posterior_samples, n_predictive, 3, test_times_x.shape[-1]+1
    ).to(device)

    simformer_predictive_samples = torch.zeros(
        num_posterior_samples, n_predictive, 3, test_times_x.shape[-1]+1
    ).to(device)

    df = pd.DataFrame(columns=["num_samples", "time_ns"])
    num_samples_list = []
    time_ns_list = []
    for i in range(n_predictive):

        print(f"Sample {i}")
        # sample from posterior
        time_start = time_ns()

        cont_samples, fin_samples = model.sample(num_posterior_samples,
                                                 ctx=test_x[i],
                                                 point_positions=test_times_theta[i]/T,
                                                 ctx_point_positions=test_times_x[i]/T
                                                )
        time_end = time_ns()
        num_samples_list.append(num_posterior_samples)
        time_ns_list.append(time_end - time_start)
        cont_samples = torch.clamp(cont_samples, min=0.0, max=1.0)
        fin_samples = torch.clamp(fin_samples, min=0.0, max=0.5)
        # cont_samples, fin_samples = rejection_sample(
        #     num_samples=num_posterior_samples,
        #     bounds_cont = [(-5,5.0) for _ in range(test_theta_cont.shape[1])],
        #     bounds_fin =  [(0.0,0.5) for _ in range(test_theta_finite.shape[1])],
        #     proposal=model,
        #     proposal_kwargs={
        #         "ctx": test_x[i],
        #         "point_positions": test_times_theta[i]/T,
        #         "ctx_point_positions": test_times_x[i]/T,
        #     },
        #     sampling_batch_size=1000
        # )

        posterior_samples[:,i,:2] = fin_samples
        posterior_samples[:,i,2:] = cont_samples.view(num_posterior_samples, test_times_theta.shape[-1])


        test_times_x_with_0 = torch.cat((torch.Tensor([0.0]).to(device), test_times_x[i]+1e-6), dim=0)

        if cfg.evaluation_config.use_simformer_initial_conditions:
            I0 = simformer_initial_conditions[i].view(-1)
            I0 = torch.clamp(I0, min=0.0)
        else:
            I0 = None



        pred = SIR(beta = cont_samples.view(num_posterior_samples, test_times_theta.shape[-1]),
                   ts = test_times_theta[i],
                   gamma = fin_samples[:,0],
                   delta = fin_samples[:,1],
                   likelihood_scale = cfg.data_config.likelihood_scale,
                   tx = test_times_x_with_0,
                   I0 = I0,
                   device=device)
        

        simformer_pred = SIR(   beta=simformer_posterior_samples[:,i,2:],
                                ts = test_times_theta[i],
                                gamma = simformer_posterior_samples[:,i,0],
                                delta = simformer_posterior_samples[:,i,1],
                                likelihood_scale = cfg.data_config.likelihood_scale,
                                tx = test_times_x_with_0,
                                I0 = I0,
                                device=device
        )

        posterior_predictive_samples[:, i, :, :] = pred
        simformer_predictive_samples[:, i, :, :] = simformer_pred

    df["num_samples"] = np.array(num_samples_list)
    df["time_ns"] = np.array(time_ns_list)
    df.to_csv(folder_path / "sampling_times.csv", index=False)

    test_theta = test_theta.cpu()
    # test_x = test_x.cpu()
    posterior_samples = posterior_samples.cpu()
    # posterior_predictive_samples = posterior_predictive_samples.cpu()

    dap_samples = posterior_samples[0, :, :]

    #We don't use run_sbc_save_results here because we have new times for every sample here.
    reduce_fns = [
        eval(f"lambda theta, x: theta[:, {i}]")
        for i in range(test_theta.shape[-1])
    ]

    ranks = _run_sbc(
        test_theta[:n_sbc],
        test_x[:n_sbc].cpu(),
        posterior_samples[:,:n_sbc],
        posterior=None,
        reduce_fns=reduce_fns, #use all marginals
    )

    check_stats = check_sbc(
        ranks,
        test_theta[:n_sbc].view(n_sbc, -1),
        dap_samples,
        num_posterior_samples=num_posterior_samples,
    )

    coverage_values = ranks / num_posterior_samples


    atcs = []
    absolute_atcs = []
    # TODO: In principle doable with torch.histrogramdd() but this is bugged right now.
    for dim_idx in range(coverage_values.shape[1]):
        # calculate empirical CDF via cumsum and normalize
        hist, alpha_grid = torch.histogram(
            coverage_values[:, dim_idx], density=True, bins=30
        )
        # add 0 to the beginning of the ecp curve to match the alpha grid
        ecp = torch.cat([torch.Tensor([0]), torch.cumsum(hist, dim=0) / hist.sum()])
        atc = (ecp - alpha_grid).mean().item()
        absolute_atc = (ecp - alpha_grid).abs().mean().item()
        atcs.append(atc)
        absolute_atcs.append(absolute_atc)

    atcs = torch.tensor(atcs)
    absolute_atcs = torch.tensor(absolute_atcs)
    print("fnope atcs: ", atcs)
    print("fnope absolute_atcs: ", absolute_atcs)

    # construct dict to save the evaluation results
    sbc_results = {
        "ranks": ranks,
        "check_stats": check_stats,
        "atcs": atcs,
        "absolute_atcs": absolute_atcs,
    }

    # save the results
    with open(os.path.join(folder_path, "fno_sbc_results.pkl"), "wb") as f:
        pickle.dump(sbc_results, f)


    #Now the same for simformer
    simformer_posterior_samples = simformer_posterior_samples.cpu()
    dap_samples = simformer_posterior_samples[0, :, :]

    ranks = _run_sbc(
        test_theta[:n_sbc],
        test_x[:n_sbc].cpu(),
        simformer_posterior_samples[:,:n_sbc],
        posterior=None,
        reduce_fns=reduce_fns, #use all marginals
    )

    check_stats = check_sbc(
        ranks,
        test_theta[:n_sbc].view(n_sbc, -1),
        dap_samples,
        num_posterior_samples=num_posterior_samples,
    )

    coverage_values = ranks / num_posterior_samples
    atcs = []
    absolute_atcs = []
    for dim_idx in range(coverage_values.shape[1]):
        hist, alpha_grid = torch.histogram(
            coverage_values[:, dim_idx], density=True, bins=30
        )
        ecp = torch.cat([torch.Tensor([0]), torch.cumsum(hist, dim=0) / hist.sum()])
        atc = (ecp - alpha_grid).mean().item()
        absolute_atc = (ecp - alpha_grid).abs().mean().item()
        atcs.append(atc)
        absolute_atcs.append(absolute_atc)

    atcs = torch.tensor(atcs)
    absolute_atcs = torch.tensor(absolute_atcs)
    print("simformer atcs: ", atcs)
    print("simformer absolute_atcs: ", absolute_atcs)

    # construct dict to save the evaluation results
    sbc_results = {
        "ranks": ranks,
        "check_stats": check_stats,
        "atcs": atcs,
        "absolute_atcs": absolute_atcs,
    }

    # save the results
    with open(os.path.join(folder_path, "simformer_sbc_results.pkl"), "wb") as f:
        pickle.dump(sbc_results, f)
    print("SBC done")



    print("Running TARP...")
    # run tarp
    n_tarp = cfg.evaluation_config.n_tarp
    reference_points = get_tarp_references(test_theta[:n_tarp].view(n_tarp, -1)).cpu()

    # posterior is not needed in _run_tarp:
    ecp, alpha_grid = _run_tarp(
        posterior_samples[:, :n_tarp],
        test_theta[:n_tarp].view(n_tarp, -1),
        references=reference_points,
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
    with open(os.path.join(folder_path, "fno_tarp_results.pkl"), "wb") as f:
        pickle.dump(tarp_results, f)

    # Now the same for simformer
    reference_points = get_tarp_references(test_theta[:n_tarp].view(n_tarp, -1)).cpu()
    ecp, alpha_grid = _run_tarp(
        simformer_posterior_samples[:, :n_tarp],
        test_theta[:n_tarp].view(n_tarp, -1),
        references=reference_points,
        distance=l2,
        z_score_theta=True,
    )
    atc, absolute_atc, kstest_pvals = check_tarp(ecp, alpha_grid)
    tarp_results = {
        "ecp": ecp,
        "alpha_grid": alpha_grid,
        "absolute_atcs": absolute_atc,
        "atcs": atc,
        "kstest_pvals": kstest_pvals,
    }
    with open(os.path.join(folder_path, "simformer_tarp_results.pkl"), "wb") as f:
        pickle.dump(tarp_results, f)

    print("TARP done")
   

    # run posterior predictive checks
    print("Running posterior predictive checks...")


    run_predictive_checks_save_results(
        test_x[:n_predictive].to(device),
        #we added time 0 to the prediction which is not in the data we condition on, so remove it
        posterior_predictive_samples[...,1:].to(device), 
        path_to_save=str(folder_path),
        prependix="fno_",
    )
    print("Posterior predictive checks done")
    print("saving...")


    predictive_summary = {
        "theta_test": test_theta[:n_predictive].detach().cpu().numpy(),
        "x_test": test_x[:n_predictive].detach().cpu().numpy(),
        "posterior_samples": posterior_samples.detach().cpu().numpy(),
        "posterior_predictive_samples": posterior_predictive_samples.detach().cpu().numpy(),
    }
    with open(
        Path(folder_path / "fno_predictive_summary.pkl"), "wb"
    ) as f:
        pickle.dump(predictive_summary, f)

    #Now the same for simformer
    run_predictive_checks_save_results(
        test_x[:n_predictive].to(device),
        #we added time 0 to the prediction which is not in the data we condition on, so remove it
        simformer_predictive_samples[...,1:].to(device),
        path_to_save=str(folder_path),
        prependix="simformer_",
    )
    print("Posterior predictive checks done")
    predictive_summary = {
        "theta_test": test_theta[:n_predictive].detach().cpu().numpy(),
        "x_test": test_x[:n_predictive].detach().cpu().numpy(),
        "posterior_samples": simformer_posterior_samples.detach().cpu().numpy(),
        "posterior_predictive_samples": simformer_predictive_samples.detach().cpu().numpy(),
    }
    with open(
        Path(folder_path / "simformer_predictive_summary.pkl"), "wb"
    ) as f:
        pickle.dump(predictive_summary, f)

    print("Completed all evaluations successfully!")


if __name__ == "__main__":
    run_fnope()



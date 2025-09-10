# This file is adapted from:
# "Simulation-Based Inference of Surface Accumulation and Basal Melt Rates of an Antarctic Ice Shelf from Isochronal Layers", https://arxiv.org/abs/2312.02997
#  https://github.com/mackelab/sbi-ice/

from fnope.simulators.ice_simulator import Layer_Tracing_Sim as lts
from fnope.simulators.ice_simulator import noise_model
from fnope.simulators.ice_simulator.modelling_utils import regrid
import torch
import pickle
from pathlib import Path
from fnope.utils.misc import get_data_dir
import os

data_dir = get_data_dir()



def simulate_one_sample(cfg, smb_x,smb_sample, true_layer, layer_mask, selection_method="advanced_noise"):
    geom = lts.Geom(nx_iso=cfg.data_config.simulator.iso_hparams.nx_iso,ny_iso=cfg.data_config.simulator.iso_hparams.ny_iso)
    x_setup,bs,ss,vxs,tmb,dQdx,dQdy = lts.init_fields_from_fname(geom,Path(data_dir/cfg.data_config.simulator.setup_file))

    smb = smb_sample.copy()
    smb = regrid(smb_x,smb,x_setup)
    smb_regrid,bmb_regrid = lts.init_mb(geom,x_setup,tmb,smb=smb)
    scheds = cfg.data_config.simulator.scheds
    init_layers = cfg.data_config.simulator.iso_hparams.init_layers
    for key in cfg.data_config.simulator.scheds.keys():
        sched = lts.Scheduele(**scheds[key])
        geom.initialize_layers(sched,init_layers)
        lts.sim(geom,smb_regrid,bmb_regrid,sched)


    #Add noise to the simulated layers
    idxs, dsum_iso, age_iso = geom.extract_nonzero_layers()
    active_trackers = geom.extract_active_trackers()
    #return all the layers, and also the layer best matching the true layer
    input_layers = torch.Tensor(geom.dsum_iso[:,0,:]+geom.bs[:,:]).T
    layers_thickness = geom.dsum_iso[:,0,:].T
    heights = layers_thickness + geom.bs.flatten()
    layer_depths = geom.ss.flatten() - heights
    layer_depths = torch.from_numpy(layer_depths)
    layer_depths = torch.flip(layer_depths,dims=(0,))
    layer_xs = torch.stack([torch.tensor(geom.x) for i in range(layer_depths.shape[0])])

    #If using the advanced noise method, load in the PSD file. Currently other noise models not implemented.
    if selection_method == "MSE":
        print("Warning: using advanced_noise method instead of MSE.")
    if selection_method == "advanced_noise":
        try:
            PSD_dict = pickle.load(Path(data_dir / "sbi_ice/PSD_matched_noise.pkl").open("rb"))
            freq = PSD_dict["freqs"][0]
            PSD_log_mean = PSD_dict["PSD_log_diffs_means"][0]
            PSD_log_var = PSD_dict["PSD_log_diffs_vars"][0]
            base_error,depth_corr,picking_error,error = noise_model.depth_error(layer_xs,layer_depths,freq,PSD_log_mean,PSD_log_var)
        except:
            print("No PSD file found! If you are using the advanced noise method, please make sure you have run the PSD matching experiment first!")
            print("Using MSE instead of advanced noise method")
            base_error,depth_corr,picking_error,error = noise_model.depth_error(layer_xs,layer_depths)
    else:
        raise NotImplementedError("method must be one of MSE or advanced_noise")
    
    #Select the best layer and return results.
    flipped_error = torch.flip(error,dims=(0,))
    input_layers = input_layers + flipped_error
    best_contour,norm,aidx = noise_model.best_contour(true_layer,input_layers,layer_mask=layer_mask,method=selection_method)
    #best_contour = modelling_utils.regrid(geom.x, best_contour.numpy(), x_eval,kind="linear")
    best_age = geom.age_iso[aidx]

    return best_contour, best_age, norm, bmb_regrid

def simulate_layers(cfg, smb_x,smb_samples, true_layer, layer_mask, path_to_save = "", selection_method="advanced_noise"):
    """
    Simulate layers using the Layer Tracing Simulator (LTS).
    Args:
        cfg: Configuration object containing simulation parameters.
        smb_samples: Samples of surface mass balance (SMB) to be used in the simulation.
        n_samples: Number of samples to simulate.
    Returns:
        bmb_list: List of calculated basal mass balance (BMB) fields.
        dsum_iso_list: List of simulated ice thickness fields.
        age_iso_list: List of simulated ice age fields.
        tracker_list: List of active trackers in the simulation.
    """

    best_layer_list = []
    bmb_list = []
    norms = []
    age_list = []
    n_samples = smb_samples.shape[0]

    for j in range(n_samples):
        print("Simulating sample %d of %d" % (j+1,n_samples))
        best_contour,best_age,norm,bmb_regrid = simulate_one_sample(cfg, smb_x,smb_samples[j], true_layer, layer_mask, selection_method=selection_method)


        best_layer_list.append(best_contour)
        bmb_list.append(torch.from_numpy(bmb_regrid).to(torch.float32))
        norms.append(norm/layer_mask.sum())
        age_list.append(best_age)

    #Convert to torch tensors and save
    bmb_out = torch.stack(bmb_list).to(torch.float32)
    best_layer_out = torch.stack(best_layer_list)
    norm_out = torch.stack(norms).to(torch.float32)
    age_out = torch.Tensor(age_list).to(torch.float32)
    print("norms",norm_out)
    res = {
        "bmbs": bmb_out,
        "best_layers": best_layer_out,
        "norms": norm_out,
        "ages": age_out
    }

    with open(os.path.join(path_to_save, "real_layers_predictive_simulations_summary.pkl"), "wb") as f:
            pickle.dump(res, f)
    
    mses = {"mses": norm_out}
    with open(os.path.join(path_to_save, "real_layers_predictive_check_results.pkl"), "wb") as f:
        pickle.dump(mses, f)
    


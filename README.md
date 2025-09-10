# Fourier Neural Operators for Posterior Estimation (FNOPE)

This repository implements FNOPE, a method for simulation-based inference for function-valued parameters. It also implements the main experiments from the [preprint](https://arxiv.org/abs/2505.22573).

## Installation

Clone and install this repository:
```
git clone https://github.com/mackelab/fnope
cd fnope
pip install -e .
```

To run the Darcy simulator and experiments, you also have to install the [physicsnemo](https://github.com/NVIDIA/physicsnemo) package. in the `fnope` environment, run:

``` 
pip install warp-lang
pip install nvidia-physicsnemo
```

**NOTE**: `physicsnemo` will install torch>=2.8.0, whereas `sbi` requires `torch<=2.6.0`. `pip` will throw a warning, but there are no breaking conflicts.

**NOTE** This packages installs a [fork](https://github.com/gmoss13/sbi) of the sbi package to fix some temporary issues with the implementation of FMPE in the previous release of `sbi`. 

This repository will be updated to match `sbi==0.25` soon.

## Usage

We use [hydra](https://hydra.cc/) for experiment management.

The experiments from the preprint are found under `fnope/experiments/`. To run the experiments, first generate the necessary data using `data_generation/simulate_{TASK_NAME}.py`. Note that data for the `ice` experiment is [publicly available](https://zenodo.org/records/10245153).

Make sure to update the root folder in `fnope/config/base_paths` to the absolute path of your current folder. Then train and evaluate FNOPE on any task with 'python fnope/experiments/{TASK_NAME}/run_fnope_{TASK_NAME}', or the baselines with 'python fnope/experiments/{TASK_NAME}/run_baseline_sbi_{TASK_NAME}'. Each run file has its own config file under `config`, which you can change to run sweeps or local debug runs.

## Tutorials

We provide two tutorial jupyter notebooks for training FNOPE models under `notebooks`.


## Citation

@article{moss2025fnope,
       author = {{Moss}, Guy and {Muhle}, Leah Sophie and {Drews}, Reinhard and {Macke}, Jakob H. and {Schr{\"o}der}, Cornelius},
        title = {FNOPE: Simulation-based inference on function spaces with Fourier Neural Operators},
      journal = {arXiv e-prints},
         year = {2025},
}


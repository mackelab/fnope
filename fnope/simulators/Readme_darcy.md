### Running the darcy simulator

The implementation is adapted from NVIDIA, using `warp` for efficient GPU usage. 
It only runs if a GPU is available. 

The torch version is in conflict with e.g. sbi torch version of the `fnope` package. 

So you need to create a seperate conda environment and install the following packages:
1. create env with python==3.12
2. install fnope
3. install:
    - `pip install warp-lang`
    - `pip install nvidia-physicsnemo`

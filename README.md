<p align="center">
  <img src="/logo_new.png" width="350" align="center">
</p>

# torchFOAM
This repository is about how to use PyTorch with OpenFOAM&reg;.

## How to run
I usually use Compute Canada clusters to run this repo. In order to use PyTorch within OpenFOAM module on one of Compute Canada clusters, you need to do the following steps:
1. Load the following modules:
+ StdEnv/2020  
+ gcc/9.3.0  
+ openmpi/4.0.3 
+ openfoam/10
2. Download the libtorch library from [here](https://pytorch.org/).
3. Unzip the downloaded file and set an environment variable TORCH_LIBRARIES to /path/to/libtorch 
4. You are good to wmake the code!



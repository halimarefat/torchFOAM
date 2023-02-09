<p align="center">
  <img src="/logo_new.png" width="350" align="center">
</p>

# torchFOAM
This repository is about how to use PyTorch with OpenFOAM&reg;.

## How to run
I usually use Compute Canada clusters to run this repo. In order to use PyTorch within OpenFOAM module on one of Compute Canada clusters, you need to do the following steps:
1. Run module load openfoam/10: note that you need to run an openFOAM package that is use C++14.
2. Download the libtorch library from [here](https://pytorch.org/).
3. Unzip the downloaded file and set an environment variable TORCH_LIBRARIES to /path/to/libtorch 
4. You are good to wmake the code!



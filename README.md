<p align="center">
  <img src="/logo_new.png" width="350" align="center">
</p>

# torchFOAM
This repository is about how to use PyTorch with OpenFOAM&reg;.

## Runing on Compute Canada clusters
Please note that due to having no access to root on clusters you may have some warnings. Therefore, it is encoraged to use containers running torchFOAM. In order to use PyTorch within OpenFOAM module on one of Compute Canada clusters, you need to do the following steps:
1. Load the following modules:
+ StdEnv/2020  
+ gcc/9.3.0  
+ openmpi/4.0.3 
+ openfoam/10
2. Download the libtorch library from [here](https://pytorch.org/).
3. Unzip the downloaded file and set an environment variable TORCH_LIBRARIES to /path/to/libtorch 
4. You are good to wmake the code!

## Running on a container
Before explaining how to use containers to run torchFOAM, I would like to explain a little bit about containers.

### Containers vs VMs

### Docker vs Apptainer

### Dockerfile step-by-step

### Using a docker container on Compute Canada clusters (or others!)

## Acknowledgment
Many thanks to Andre Weiner for his tutorials on this topic.

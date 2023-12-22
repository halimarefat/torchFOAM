#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=0-15:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=NN_M4_103

module restore torchfoamenv
blockMesh
topoSet
decomposePar
srun /home/hmarefat/projects/def-alamj/shared/bin/v6/atmosphericLES -parallel > casec_log.out
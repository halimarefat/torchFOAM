#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#SBATCH --time=0-16:00:00
#SBATCH --mem-per-cpu=800
#SBATCH --job-name=dynSmagCase

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 cudnn openmpi/4.0.3 openfoam/6
blockMesh
topoSet
decomposePar
srun /home/hmarefat/projects/def-alamj/shared/bin/v6/atmosphericLES -parallel > casec_log.out
reconstructPar 



#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=0-23:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --job-name=NN_M4_503

module restore torchfoamenv
srun /home/hmarefat/projects/def-alamj/shared/bin/v6/atmosphericLES -parallel > casec_log.out
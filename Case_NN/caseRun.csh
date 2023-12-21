#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=nnCase

module restore torchfoamenv
srun /home/hmarefat/projects/def-alamj/shared/bin/v6/atmosphericLES -parallel > casec_log_1.out
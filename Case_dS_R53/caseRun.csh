#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#SBATCH --time=0-12:00:00
#SBATCH --mem-per-cpu=800
#SBATCH --job-name=Case_dS_R53

module restore torchfoamenv
srun /home/hmarefat/projects/def-alamj/shared/bin/v6/atmosphericLES -parallel > casec_log.out
reconstructPar 


#!/bin/bash

#SBATCH --account=def-alamj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=0-4:00:00
#SBATCH --mem-per-cpu=800
#SBATCH --job-name=SACase

module restore torchfoamenv
blockMesh
topoSet 
decomposePar 
srun /home/hmarefat/projects/def-alamj/shared/bin/v6/atmosphericLES -parallel > casec_log.out



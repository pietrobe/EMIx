#!/bin/bash

#SBATCH --job-name=emi_test
#SBATCH --account=NN8049K
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2G

## Set up job environment
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default


# for FEniCS
# module load matplotlib/3.0.0-intel-2018b-Python-3.6.6

# Source common FEniCS installation
# source /cluster/shared/fenics/conf/fenics-2019.1.0.saga.intel.conf
# Source FEniCS installation with newer scipy
# source /cluster/shared/fenics/conf/dolfin-adjoint_for_fenics-2019.1.0.saga.intel.conf
source /cluster/shared/fenics/fenics-2019.2.0.dev0.saga.intel-2020a-py3.8.conf 

dijitso clean

## Do some work:
# python heat.py

## try in parallel 
srun python ../src/emi.py

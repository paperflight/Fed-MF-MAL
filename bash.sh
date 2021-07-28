#!/bin/bash -l                                                                                    
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --job-name=alphavr
# #SBATCH --gres=gpu
#SBATCH --ntasks=6
#SBATCH --mem=20000
#SBATCH --time=0-10:00
# #SBATCH --constrain=v100
#SBATCH --constrain=skylake

ulimit -n 4096
git status
cat GLOBAL_PRARM.py

# module load libs/cuda
python ./train.py --id='default' --previous-action-observable --architecture='canonical_61obv_16ap'

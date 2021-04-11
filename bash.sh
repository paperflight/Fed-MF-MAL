#!/bin/bash -l                                                                                    
#SBATCH --output=/mnt/lustre/users/%u/%j.out
#SBATCH --job-name=alphavr
# #SBATCH --gres=gpu
#SBATCH --ntasks=16
#SBATCH --mem=40000
#SBATCH --time=0-72:00
# #SBATCH --constrain=v100
#SBATCH --constrain=skylake

ulimit -n 4096
git status
cat GLOBAL_PRARM.py

# module load libs/cuda
python ./train.py

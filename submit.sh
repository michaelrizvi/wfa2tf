#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for CPUs
#SBATCH --gres=gpu:1                                     # Ask for GPUs
#SBATCH --mem=2G                                        # Ask for RAM
#SBATCH --time=3:00:00                                   # Job runtime
​

# create a local virtual environnement (on the compute node)
module load python/3.8
source ~/test-env/bin/activate
python train_counting.py


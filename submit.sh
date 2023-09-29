#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                                   # The job will run for 3 hours

# create a local virtual environnement (on the compute node)
module load python/3.8
source ~/test-env/bin/activate

python train_counting.py --nlayers 9 --seqlen=16
python train_counting.py --nlayers 10 --seqlen=16
python train_counting.py --nlayers 11 --seqlen=16
python train_counting.py --nlayers 12 --seqlen=16

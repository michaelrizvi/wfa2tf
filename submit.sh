#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                                   # The job will run for 3 hours

# create a local virtual environnement (on the compute node)
module load python/3.8
source ~/test-env/bin/activate

python train_counting.py --batchsize 8 --dropout 0.2 --lr 0.001 --nlayers 1
python train_counting.py --batchsize 16 --dropout 0.2 --lr 0.001 --nlayers 1
python train_counting.py --batchsize 8 --dropout 0.2 --lr 0.001 --nlayers 2
python train_counting.py --batchsize 16 --dropout 0.2 --lr 0.001 --nlayers 2
python train_counting.py --batchsize 8 --dropout 0.2 --lr 0.001 --nlayers 3
python train_counting.py --batchsize 16 --dropout 0.2 --lr 0.001 --nlayers 3
python train_counting.py --batchsize 8 --dropout 0.2 --lr 0.001 --nlayers 4
python train_counting.py --batchsize 16 --dropout 0.2 --lr 0.001 --nlayers 4
python train_counting.py --batchsize 8 --dropout 0.2 --lr 0.001 --nlayers 5
python train_counting.py --batchsize 16 --dropout 0.2 --lr 0.001 --nlayers 5
python train_counting.py --batchsize 8 --dropout 0.2 --lr 0.001 --nlayers 6
python train_counting.py --batchsize 16 --dropout 0.2 --lr 0.001 --nlayers 6
python train_counting.py --batchsize 8 --dropout 0.2 --lr 0.001 --nlayers 7
python train_counting.py --batchsize 16 --dropout 0.2 --lr 0.001 --nlayers 7

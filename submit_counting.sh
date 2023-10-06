#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:4                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                                   # The job will run for 3 hours

# create a local virtual environnement (on the compute node)
module load libffi
module load python/3.8
source ~/test-env/bin/activate

python train_counting.py --k=2 --nbL=10  --seqlen=32 --emsize=2 --d_hid=2 
python train_counting.py --k=4 --nbL=10  --seqlen=32 --emsize=2 --d_hid=2 
python train_counting.py --k=6 --nbL=10  --seqlen=32 --emsize=2 --d_hid=2 
python train_counting.py --k=8 --nbL=10  --seqlen=32 --emsize=2 --d_hid=2 
# python3 train_counting_reps.py --epochs=100 --batchsize=10 --seqlen=8 --lr=0.001 --ntry=10 --debug=True

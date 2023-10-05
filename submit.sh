#!/bin/sh
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                                   # The job will run for 3 hours
#

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "No command-line argument provided."
    exit 1
fi

# Get and print the first command-line argument
arg="$1"
echo "Automata number: $arg"

# create a local virtual environnement (on the compute node)
module load python/3.8
source ~/test-env/bin/activate

python train_pautomac_reps.py --nlayers "$1" --seqlen 16 --ntry 10 --nb_aut 39 --emsize 144 --d_hid 36
python train_pautomac_reps.py --nlayers "$1" --seqlen 16 --ntry 10 --nb_aut 46 --emsize 362 --d_hid 362 
python train_pautomac_reps.py --nlayers "$1" --seqlen 32 --ntry 10 --nb_aut 39 --emsize 144 --d_hid 36 
python train_pautomac_reps.py --nlayers "$1" --seqlen 32 --ntry 10 --nb_aut 46 --emsize 362 --d_hid 362 


# python3 train_counting_reps.py --epochs=100 --batchsize=10 --seqlen=8 --lr=0.001 --ntry=10 --debug=True

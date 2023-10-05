#!/bin/bash
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                                   # The job will run for 3 hours
# Check if at least one argument is provided

if [ $# -eq 0 ]; then
    echo "No command-line argument provided."
    exit 1
fi

# List of integers
#integers=(4 12 14 20 30 31 33 38 39 45 46)
integers=(4 46)

# Path to the Python script

# Iterate over the list of integers
for integer in "${integers[@]}"; do
    echo "Processing automata: $integer"
    # Run the Python script with the integer as a command-line argument
    python pautomac_generation.py --nb_aut="$integer" --nbEx=10000 --seqlen="$1"
done

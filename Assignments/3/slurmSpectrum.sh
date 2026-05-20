#!/bin/bash

#SBATCH --job-name=pi-a3
#SBATCH --output=spectrum.out
#SBATCH --error=spectrum.err
#SBATCH --nodes=3
#SBATCH --ntasks=12
#SBATCH --gres=gpu:4
#SBATCH --partition=el8-rpi
#SBATCH --time=00:30:00

module load xl_r spectrum-mpi cuda/11.2

# For some reason this wouldn't create the directory on its own idk why
# mkdir -p /data 

echo "Weak scaling"
for ranks in 1 2 4 6 8 12; do
    export SAMPLES_PER_RANK=$((8 * 2**30)) # 8 billion samples
    echo "Running with $ranks ranks"
    mpirun -np $ranks ./A3 $SAMPLES_PER_RANK > data/weak_${ranks}.txt
done

echo "Strong scaling"
for ranks in 1 2 4 6 8 12; do
    export SAMPLES_PER_RANK=$((96 * 2**30 / ranks)) # 96 billion samples
    echo "Running with $ranks ranks"
    mpirun -np $ranks ./A3 $SAMPLES_PER_RANK > data/strong_${ranks}.txt
done

echo "Done."
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

# --- Weak Scaling ---
echo "Starting Weak Scaling Performance Study..."
echo "Threads\tTime (seconds)"

# Loop through different thread counts for weak scaling
for threads in 1 2 4 8 16 32 64 128; do
  echo -n "$threads\t"
  time ./pthread-compute-pi $threads 0 # 0 for weak scaling
  echo ""
done

echo "Weak Scaling Complete."

# --- Strong Scaling ---
echo "Starting Strong Scaling Performance Study..."
echo "Threads\tTime (seconds)"

# Loop through different thread counts for strong scaling
for threads in 1 2 4 8 16 32 64 128; do
  echo -n "$threads\t"
  time ./pthread-compute-pi $threads 1 # 1 for strong scaling
  echo ""
done

echo "Strong Scaling Complete."

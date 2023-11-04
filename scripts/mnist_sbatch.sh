#!/bin/bash
#SBATCH -J mnist
#SBATCH -n 2  # Number of tasks (adjust as needed)
#SBATCH -o output.log
#SBATCH -e error.log
#SBATCH --nodes 2

set -ex

# Execute integration tests.
srun --no-kill /fsx/scripts/mnist_launch.sh

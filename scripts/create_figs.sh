#!/bin/bash
#SBATCH --job-name=scaling_baseline
#SBATCH --partition=normal
#SBATCH --account=imacm
#SBATCH -N 1                      # Number of nodes
#SBATCH --ntasks=1                # Total number of tasks (i.e., parallel jobs).
#SBATCH --cpus-per-task=8         # Number of CPUs per task.
#SBATCH --mem-per-cpu=4096 # in MB
#SBATCH --time=08:00:00

# --- Module Management ---
# Unload Anaconda3 to prevent conflicts.
module unload Anaconda3/2022.05

# Load Necessary Modules.
module load 2022a GCCcore/11.3.0 Python/3.10.4

# --- Run Parallel Jobs ---
srun python results.py
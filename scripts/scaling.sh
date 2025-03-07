#!/bin/bash
#SBATCH --job-name=scaling_baseline
#SBATCH --partition=normal
#SBATCH --account=imacm
#SBATCH -N 1                      # Number of nodes
#SBATCH --ntasks=1                # Total number of tasks (i.e., parallel jobs).
#SBATCH --cpus-per-task=1         # Number of CPUs per task.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Set the number of training trajectories.
N=$1
EPOCHS=$2
REPEAT=$3
METHOD=$4
NAME=$5
HIDDEN_DIM=$6

# shellcheck disable=SC2050
if [[ $METHOD == "baseline" ]]; then
  experiment="scaling_initial_baseline"
  baseline_flag="--baseline"
  affine_flag=""
elif [[ $METHOD == "affine" ]]; then
  experiment="scaling_initial_affine"
  baseline_flag=""
  affine_flag="--affine"
else
  experiment="scaling_initial_default"
  baseline_flag=""
  affine_flag=""
fi

# Append NAME to the experiment flag.
experiment="${experiment}_${NAME}"

# --- Module Management ---
# Unload Anaconda3 to prevent conflicts.
module unload Anaconda3/2022.05

# Load Necessary Modules.
module load 2022a GCCcore/11.3.0 Python/3.10.4

# --- Run Parallel Jobs ---
srun --exclusive -n1 python main.py --name "$NAME" --num_trajectories "$N" --num_val_trajectories 1000 --lr 1e-3 --epochs "$EPOCHS"  --repeat "$REPEAT"  --J default --R default --G mlp --output-weight .25 --hidden_dim "$HIDDEN_DIM" --experiment ${experiment} ${baseline_flag} ${affine_flag} --no-forecast

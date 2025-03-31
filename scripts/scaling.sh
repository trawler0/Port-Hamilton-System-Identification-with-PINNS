#!/bin/bash
#SBATCH --job-name=scaling_baseline
#SBATCH --partition=normal
#SBATCH --account=imacm
#SBATCH -N 1                      # Number of nodes
#SBATCH --ntasks=1                # Total number of tasks (i.e., parallel jobs).
#SBATCH --cpus-per-task=8         # Number of CPUs per task.
#SBATCH --mem-per-cpu=4096 # in MB
#SBATCH --time=0-08:00:00
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
  experiment="scaling_baseline"
  baseline_flag="--baseline"
  affine_flag=""
  J="none"
  R="none"
  G="none"
  grad_H="none"
elif [[ $METHOD == "affine" ]]; then
  experiment="scaling_affine"
  baseline_flag=""
  affine_flag="--affine"
  J="none"
  R="none"
  G="none"
  grad_H="none"
elif [[ $METHOD == "spring_chain" ]]; then
  experiment="scaling_spring_chain"
  baseline_flag=""
  affine_flag=""
  J="spring_chain"
  R="spring_chain"
  G="spring_chain"
  grad_H="spring_chain"
else
  experiment="scaling_default"
  baseline_flag=""
  affine_flag=""
  J="default"
  R="default"
  G="mlp"
  grad_H="gradient"
fi

# Append NAME to the experiment flag.
experiment="${experiment}_${NAME}"

# --- Module Management ---
# Unload Anaconda3 to prevent conflicts.
module unload Anaconda3/2022.05

# Load Necessary Modules.
module load 2022a GCCcore/11.3.0 Python/3.10.4

# --- Run Parallel Jobs ---
srun --exclusive -n1 python main.py --name "$NAME" --num_trajectories "$N" --time 10 --num_val_trajectories 200 --val_time 200 --lr 1e-3 --epochs "$EPOCHS"  --repeat "$REPEAT"  --J "$J" --R "$R" --G "$G" --grad_H "$grad_H" --output-weight .25 --hidden_dim "$HIDDEN_DIM" --experiment ${experiment} ${baseline_flag} ${affine_flag} --no-forecast

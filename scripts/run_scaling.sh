#!/bin/bash

name=$1
if [[ $name == "spring_multi_mass" ]]; then
  modes="spring_chain"
elif [[ $name == "ball" ]]; then
  modes="affine default"
elif [[ $name == "motor" ]]; then
  modes="baseline affine"
elif [[ $name == "spring" ]]; then
  modes="default affine baseline"
fi

for trajectories in 2 4 8 16 32 64 128 256; do
  # Determine the walltime based on the trajectories value.
  if [ "$trajectories" -eq 2 ]; then
    time_str="02:00:00"
  elif [ "$trajectories" -eq 4 ]; then
    time_str="04:00:00"
  elif [ "$trajectories" -eq 8 ]; then
    time_str="06:00:00"
  elif [ "$trajectories" -eq 16 ]; then
    time_str="10:00:00"
  elif [ "$trajectories" -eq 32 ]; then
    time_str="18:00:00"
  elif [ "$trajectories" -eq 64 ]; then
    time_str="34:00:00"
  elif [ "$trajectories" -eq 128 ]; then
    time_str="66:00:00"
  elif [ "$trajectories" -eq 256 ]; then
    time_str="130:00:00"
  fi

  for epochs in 1 3 10 30 100 300 1000 3000 10000 30000; do
    for hidden_dim in 4 6 8 12 16 24; do
      for mode in $modes; do
        sbatch --time "$time_str" scripts/scaling.sh "$trajectories" "$epochs" 1 "$mode" "$name" "$hidden_dim"
      done
    done
  done
done

#!/bin/bash

name=$1

for trajectories in 1 3 10 30 100; do
  # Determine the walltime based on the trajectories value.
  if [ "$trajectories" -eq 1 ]; then
    time_str="00:30:00"
  elif [ "$trajectories" -eq 3 ]; then
    time_str="03:00:00"
  elif [ "$trajectories" -eq 10 ]; then
    time_str="08:00:00"
  elif [ "$trajectories" -eq 30 ]; then
    time_str="20:00:00"
  elif [ "$trajectories" -eq 100 ]; then
    time_str="40:00:00"
  fi

  for epochs in 1 3 10 30 100 300 1000 3000 10000 30000; do
    for hidden_dim in 4 6 9 16 24 32; do
      for mode in default baseline affine; do
        sbatch --time "$time_str" scripts/scaling.sh "$trajectories" "$epochs" 1 "$mode" "$name" "$hidden_dim"
      done
    done
  done
done

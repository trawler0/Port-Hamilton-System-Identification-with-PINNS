python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J default  --R default  --G mlp --output-weight .0 --run_name default_0 --experiment beta
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name _025 --experiment beta
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J   --R   --G mlp --output-weight .5 --run_name _05 --experiment beta
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J default  --R default  --G mlp --output-weight .1 --run_name default_1 --experiment beta



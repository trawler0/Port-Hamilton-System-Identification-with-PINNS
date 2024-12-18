python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .0 --run_name matmul_0 --experiment beta
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name matmul_025 --experiment beta
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .5 --run_name matmul_05 --experiment beta
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .1 --run_name matmul_1 --experiment beta



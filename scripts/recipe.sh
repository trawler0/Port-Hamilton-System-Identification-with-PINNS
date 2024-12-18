python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name default --experiment recipe
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 10 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name shorter_x.1 --experiment recipe
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name shallow --experiment recipe --depth 1





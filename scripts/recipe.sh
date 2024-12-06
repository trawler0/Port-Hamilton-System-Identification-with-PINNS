python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name default --experiment recipe
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name shorter_x.1 --experiment recipe
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name shallow --experiment recipe --depth 1





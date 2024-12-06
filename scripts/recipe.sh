python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name default --experiment recipe
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name shorter_x.2 --experiment recipe
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name no_weight_decay --experiment recipe --weigh_decay 0
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name shallow --experiment recipe --depth 1
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name narrow --experiment recipe --hidden_dim 16





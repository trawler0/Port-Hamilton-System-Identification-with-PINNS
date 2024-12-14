# noise
python main.py --name motor --num_trajectories 5 --num_val_trajectories 1000 --lr 1e-3 --epochs 2500 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise --experiment noise --dB 20
python main.py --name motor --num_trajectories 25 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise --experiment noise --dB 20
python main.py --name motor --num_trajectories 125 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise --experiment noise --dB 20

# no noise
python main.py --name motor --num_trajectories 5 --num_val_trajectories 1000 --lr 1e-3 --epochs 2500 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name no_noise --experiment noise
python main.py --name motor --num_trajectories 25 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name no_noise --experiment noise
python main.py --name motor --num_trajectories 125 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name no_noise --experiment noise

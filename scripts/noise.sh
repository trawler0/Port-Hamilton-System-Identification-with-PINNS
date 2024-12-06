# matmul
python main.py --name motor --num_trajectories 5 --num_val_trajectories 1000 --lr 1e-3 --epochs 5000 --repeat 100 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_40 --experiment noise --dB 60
python main.py --name motor --num_trajectories 5 --num_val_trajectories 1000 --lr 1e-3 --epochs 5000 --repeat 100 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_30 --experiment noise --dB 40
python main.py --name motor --num_trajectories 5 --num_val_trajectories 1000 --lr 1e-3 --epochs 5000 --repeat 100 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_20 --experiment noise --dB 20
python main.py --name motor --num_trajectories 5 --num_val_trajectories 1000 --lr 1e-3 --epochs 5000 --repeat 100 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_10 --experiment noise --dB 10

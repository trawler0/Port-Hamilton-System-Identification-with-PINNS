# matmul
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_60 --experiment noise --dB 60
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_40 --experiment noise --dB 40
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_20 --experiment noise --dB 20
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name noise_10 --experiment noise --dB 10

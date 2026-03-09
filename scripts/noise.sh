# noise
python main.py --name motor --num_trajectories 10 --num_val_trajectories 100 --lr 1e-3 --epochs 8000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise_6
python main.py --name motor --num_trajectories 10 --num_val_trajectories 100 --lr 1e-3 --epochs 8000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise30 --experiment noise_6 --dB 30
python main.py --name motor --num_trajectories 10 --num_val_trajectories 100 --lr 1e-3 --epochs 8000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise25 --experiment noise_6 --dB 25
python main.py --name motor --num_trajectories 10 --num_val_trajectories 100 --lr 1e-3 --epochs 8000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise35 --experiment noise_6 --dB 35

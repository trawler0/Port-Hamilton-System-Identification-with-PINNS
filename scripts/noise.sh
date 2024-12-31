# noise
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name noise30 --experiment noise --dB 30

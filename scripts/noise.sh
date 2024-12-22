# noise
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 2000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name noise --experiment noise --dB 20
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 1000 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name noise --experiment noise --dB 20
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 400 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise --experiment noise --dB 20

# no noise
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 2000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 1000 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 400 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise

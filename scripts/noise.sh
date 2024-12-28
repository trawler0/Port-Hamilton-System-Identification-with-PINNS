# noise
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name noise20 --experiment noise --dB 20
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name noise20 --experiment noise --dB 20
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise20 --experiment noise --dB 20

python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name noise30 --experiment noise --dB 30
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name noise30 --experiment noise --dB 30
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise30 --experiment noise --dB 30

python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name noise40 --experiment noise --dB 40
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name noise40 --experiment noise --dB 40
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise40 --experiment noise --dB 40

python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name noise50 --experiment noise --dB 50
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name noise50 --experiment noise --dB 50
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name noise50 --experiment noise --dB 50

# no noise
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name no_noise --experiment noise

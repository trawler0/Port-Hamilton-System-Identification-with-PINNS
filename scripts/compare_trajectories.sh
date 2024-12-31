python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name baseline --experiment compare --baseline
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment compare
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name prior --experiment compare




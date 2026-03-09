python main.py --name ball --num_trajectories 32 --num_val_trajectories 100 --lr 1e-3 --epochs 8000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name baseline --experiment compare_6 --baseline
python main.py --name ball --num_trajectories 32 --num_val_trajectories 100 --lr 1e-3 --epochs 8000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment compare_6
python main.py --name ball --num_trajectories 32 --num_val_trajectories 100 --lr 1e-3 --epochs 8000 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name prior --experiment compare_6 --no-normalize-x



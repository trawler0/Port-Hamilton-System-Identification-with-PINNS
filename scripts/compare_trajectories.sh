python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name baseline --experiment compare --baseline
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name matmul --experiment compare
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 5 --J linear  --R matmul  --G linear --output-weight .25 --run_name prior --experiment compare




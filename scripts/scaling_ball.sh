######################ball######################
python main.py --name ball --num_trajectories 2 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_2 --experiment scaling_ball_1 --baseline
python main.py --name ball --num_trajectories 2 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_2 --experiment scaling_ball_1
python main.py --name ball --num_trajectories 2 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_2 --experiment scaling_ball_1 --weight_decay 1e-3 --no-normalize-u --no-normalize-x

python main.py --name ball --num_trajectories 4 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_4 --experiment scaling_ball_1 --baseline
python main.py --name ball --num_trajectories 4 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_4 --experiment scaling_ball_1
python main.py --name ball --num_trajectories 4 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_4 --experiment scaling_ball_1 --weight_decay 1e-3 --no-normalize-u --no-normalize-x

python main.py --name ball --num_trajectories 8 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_8 --experiment scaling_ball_1 --baseline
python main.py --name ball --num_trajectories 8 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_8 --experiment scaling_ball_1
python main.py --name ball --num_trajectories 8 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_8 --experiment scaling_ball_1 --weight_decay 1e-3 --no-normalize-u --no-normalize-x

python main.py --name ball --num_trajectories 16 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_16 --experiment scaling_ball_1 --baseline
python main.py --name ball --num_trajectories 16 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_16 --experiment scaling_ball_1
python main.py --name ball --num_trajectories 16 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_16 --experiment scaling_ball_1 --weight_decay 1e-3 --no-normalize-u --no-normalize-x

python main.py --name ball --num_trajectories 32 --num_val_trajectories 100 --lr 2e-3 --epochs 5000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_32 --experiment scaling_ball_1 --baseline
python main.py --name ball --num_trajectories 32 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_32 --experiment scaling_ball_1
python main.py --name ball --num_trajectories 32 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_32 --experiment scaling_ball_1 --weight_decay 1e-3 --no-normalize-u --no-normalize-x

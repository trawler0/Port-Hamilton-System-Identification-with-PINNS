######################motor######################
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_10 --experiment scaling_motor --baseline
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_10 --experiment scaling_motor
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 2000 --repeat 10 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_10 --experiment scaling_motor --weight_decay 1e-3

python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_30 --experiment scaling_motor --baseline
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_30 --experiment scaling_motor
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 5 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_30 --experiment scaling_motor --weight_decay 1e-3

python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_100 --experiment scaling_motor --baseline
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_100 --experiment scaling_motor
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 400 --repeat 1 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_100 --experiment scaling_motor --weight_decay 1e-3

python main.py --name motor --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_300 --experiment scaling_motor --baseline
python main.py --name motor --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_300 --experiment scaling_motor
python main.py --name motor --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_300 --experiment scaling_motor --weight_decay 1e-3

python main.py --name motor --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_1000 --experiment scaling_motor --baseline
python main.py --name motor --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_1000 --experiment scaling_motor
python main.py --name motor --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_1000 --experiment scaling_motor --weight_decay 1e-3

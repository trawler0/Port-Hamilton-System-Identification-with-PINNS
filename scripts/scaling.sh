######################ball######################
# baseline
python main.py --name ball --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_1 --experiment scaling --baseline
python main.py --name ball --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_3 --experiment scaling --baseline
python main.py --name ball --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_10 --experiment scaling --baseline
python main.py --name ball --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_30 --experiment scaling --baseline
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_100 --experiment scaling --baseline
python main.py --name ball --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_300 --experiment scaling --baseline
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_1000 --experiment scaling --baseline

# default
python main.py --name ball --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_1 --experiment scaling
python main.py --name ball --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_3 --experiment scaling
python main.py --name ball --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_10 --experiment scaling
python main.py --name ball --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_30 --experiment scaling
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_100 --experiment scaling
python main.py --name ball --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_300 --experiment scaling
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball_default_1000 --experiment scaling

# prior
python main.py --name ball --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_1 --experiment scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_3 --experiment scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_10 --experiment scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_30 --experiment scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_100 --experiment scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_300 --experiment scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J linear  --R default  --G linear --output-weight .25 --run_name ball_prior_1000 --experiment scaling --weight_decay 1e-3

######################spring######################
# baseline
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1 --experiment scaling --baseline
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_3 --experiment scaling --baseline
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_10 --experiment scaling --baseline
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_30 --experiment scaling --baseline
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_100 --experiment scaling --baseline
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_300 --experiment scaling --baseline
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1000 --experiment scaling --baseline


# default
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_1 --experiment scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_3 --experiment scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_10 --experiment scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_30 --experiment scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_100 --experiment scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_300 --experiment scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_1000 --experiment scaling

# prior
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1 --experiment scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_3 --experiment scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_10 --experiment scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_30 --experiment scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_100 --experiment scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_300 --experiment scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1000 --experiment scaling --weight_decay 1e-3

######################motor######################
# baseline
python main.py --name motor --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_1 --experiment scaling --baseline
python main.py --name motor --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_3 --experiment scaling --baseline
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_10 --experiment scaling --baseline
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_30 --experiment scaling --baseline
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_100 --experiment scaling --baseline
python main.py --name motor --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_300 --experiment scaling --baseline
python main.py --name motor --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name motor_baseline_1000 --experiment scaling --baseline

# default
python main.py --name motor --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_1 --experiment scaling
python main.py --name motor --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_3 --experiment scaling
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_10 --experiment scaling
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_30 --experiment scaling
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_100 --experiment scaling
python main.py --name motor --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_300 --experiment scaling
python main.py --name motor --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name motor_default_1000 --experiment scaling

# prior
python main.py --name motor --num_trajectories 1 --num_val_trajectories 1000 --lr 2e-3 --epochs 5000 --repeat 50 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_1 --experiment scaling --weight_decay 1e-3
python main.py --name motor --num_trajectories 3 --num_val_trajectories 1000 --lr 2e-3 --epochs 1500 --repeat 20 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_3 --experiment scaling --weight_decay 1e-3
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 2e-3 --epochs 500 --repeat 10 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_10 --experiment scaling --weight_decay 1e-3
python main.py --name motor --num_trajectories 30 --num_val_trajectories 1000 --lr 2e-3 --epochs 250 --repeat 5 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_30 --experiment scaling --weight_decay 1e-3
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_100 --experiment scaling --weight_decay 1e-3
python main.py --name motor --num_trajectories 300 --num_val_trajectories 1000 --lr 2e-3 --epochs 50 --repeat 1 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_300 --experiment scaling --weight_decay 1e-3
python main.py --name motor --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 25 --repeat 1 --J default  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_prior_1000 --experiment scaling --weight_decay 1e-3

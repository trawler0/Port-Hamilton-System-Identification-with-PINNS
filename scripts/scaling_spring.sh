######################spring######################
# baseline
#python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 1e-3 --epochs 10000 --repeat 50 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1 --experiment scaling_spring --baseline
#python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 1e-3 --epochs 3000 --repeat 20 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_3 --experiment scaling_spring --baseline
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_10 --experiment scaling_spring --baseline
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_30 --experiment scaling_spring --baseline
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_100 --experiment scaling_spring --baseline
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_300 --experiment scaling_spring --baseline
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1000 --experiment scaling_spring --baseline

# default
#python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 1e-3 --epochs 10000 --repeat 50 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_1 --experiment scaling_spring
#python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 1e-3 --epochs 3000 --repeat 20 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_3 --experiment scaling_spring
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_10 --experiment scaling_spring
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_30 --experiment scaling_spring
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_100 --experiment scaling_spring
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_300 --experiment scaling_spring
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name spring_default_1000 --experiment scaling_spring

# prior
#python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 1e-3 --epochs 20000 --repeat 50 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1 --experiment scaling_spring --weight_decay 1e-3
#python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 1e-3 --epochs 6000 --repeat 20 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_3 --experiment scaling_spring --weight_decay 1e-3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 2000 --repeat 10 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_10 --experiment scaling_spring --weight_decay 1e-3
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 5 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_30 --experiment scaling_spring --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 400 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_100 --experiment scaling_spring --weight_decay 1e-3
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_300 --experiment scaling_spring --weight_decay 1e-3
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1000 --experiment scaling_spring --weight_decay 1e-3


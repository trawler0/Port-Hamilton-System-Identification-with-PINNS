# baseline
python main.py --name spring --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1 --tag scaling --baseline
python main.py --name spring --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_3 --tag scaling --baseline
python main.py --name spring --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_10 --tag scaling --baseline
python main.py --name spring --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_30 --tag scaling --baseline
python main.py --name spring --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_100 --tag scaling --baseline
python main.py --name spring --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_300 --tag scaling --baseline
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1000 --tag scaling --baseline
python main.py --name spring --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_3000 --tag scaling --baseline

# sigmoid
python main.py --name spring --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_1 --tag scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_3 --tag scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_10 --tag scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_30 --tag scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_100 --tag scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_300 --tag scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_1000 --tag scaling
python main.py --name spring --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_3000 --tag scaling

# matmul
python main.py --name spring --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_1 --tag scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_3 --tag scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_10 --tag scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_30 --tag scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_100 --tag scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_300 --tag scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_1000 --tag scaling
python main.py --name spring --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_3000 --tag scaling

# prior
python main.py --name spring --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1 --tag scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_3 --tag scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_10 --tag scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_30 --tag scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_100 --tag scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_300 --tag scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1000 --tag scaling --weight_decay 1e-3
python main.py --name spring --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_3000 --tag scaling --weight_decay 1e-3

# baseline
python main.py --name ball --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_1 --tag scaling --baseline
python main.py --name ball --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_3 --tag scaling --baseline
python main.py --name ball --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_10 --tag scaling --baseline
python main.py --name ball --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_30 --tag scaling --baseline
python main.py --name ball --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_100 --tag scaling --baseline
python main.py --name ball --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_300 --tag scaling --baseline
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_1000 --tag scaling --baseline
python main.py --name ball --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name ball_baseline_3000 --tag scaling --baseline

# sigmoid
python main.py --name ball --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_1 --tag scaling
python main.py --name ball --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_3 --tag scaling
python main.py --name ball --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_10 --tag scaling
python main.py --name ball --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_30 --tag scaling
python main.py --name ball --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_100 --tag scaling
python main.py --name ball --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_300 --tag scaling
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_1000 --tag scaling
python main.py --name ball --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_3000 --tag scaling

# matmul
python main.py --name ball --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_1 --tag scaling
python main.py --name ball --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_3 --tag scaling
python main.py --name ball --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_10 --tag scaling
python main.py --name ball --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_30 --tag scaling
python main.py --name ball --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_100 --tag scaling
python main.py --name ball --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_300 --tag scaling
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_1000 --tag scaling
python main.py --name ball --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_3000 --tag scaling

# prior
python main.py --name ball --num_trajectories 1 --num_val_trajectories 2500 --lr 1e-3 --epochs 2000 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_1 --tag scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 3 --num_val_trajectories 2500 --lr 1e-3 --epochs 600 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_3 --tag scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 10 --num_val_trajectories 2500 --lr 1e-3 --epochs 200 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_10 --tag scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 30 --num_val_trajectories 2500 --lr 1e-3 --epochs 100 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_30 --tag scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 100 --num_val_trajectories 2500 --lr 1e-3 --epochs 50 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_100 --tag scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 300 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_300 --tag scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 2500 --lr 1e-3 --epochs 20 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_1000 --tag scaling --weight_decay 1e-3
python main.py --name ball --num_trajectories 3000 --num_val_trajectories 2500 --lr 1e-3 --epochs 10 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name ball_prior_3000 --tag scaling --weight_decay 1e-3

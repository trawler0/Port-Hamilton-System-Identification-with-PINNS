# baseline
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 5e-3 --epochs 2000 --repeat 20 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1 --exp_type scaling --baseline
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 5e-3 --epochs 600 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_3 --exp_type scaling --baseline
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 5e-3 --epochs 200 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_10 --exp_type scaling --baseline
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 5e-3 --epochs 100 --repeat 5 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_30 --exp_type scaling --baseline
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_100 --exp_type scaling --baseline
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_300 --exp_type scaling --baseline
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J none  --R none  --G none --grad_H none --output-weight .25 --run_name spring_baseline_1000 --exp_type scaling --baseline

# sigmoid
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 5e-3 --epochs 2000 --repeat 20 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_1 --exp_type scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 5e-3 --epochs 600 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_3 --exp_type scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 5e-3 --epochs 200 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_10 --exp_type scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 5e-3 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_30 --exp_type scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_100 --exp_type scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_300 --exp_type scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_1000 --exp_type scaling

# matmul
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 5e-3 --epochs 2000 --repeat 20 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_1 --exp_type scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 5e-3 --epochs 600 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_3 --exp_type scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 5e-3 --epochs 200 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_10 --exp_type scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 5e-3 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_30 --exp_type scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_100 --exp_type scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_300 --exp_type scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_1000 --exp_type scaling

# prior
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 5e-3 --epochs 2000 --repeat 20 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1 --exp_type scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 5e-3 --epochs 600 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_3 --exp_type scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 5e-3 --epochs 200 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_10 --exp_type scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 5e-3 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_30 --exp_type scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_100 --exp_type scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_300 --exp_type scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 5e-3 --epochs 20 --repeat 1 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_prior_1000 --exp_type scaling

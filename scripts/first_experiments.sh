# spring
#python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid_sigmoid
#python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_matmul
#python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J linear  --R matmul  --G linear --grad_H linear --output-weight .25 --run_name spring_linear_matmul

# motor
#python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name motor_sigmoid_sigmoid
#python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name motor_matmul_matmul

# ball
#python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_sigmoid
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_matmul

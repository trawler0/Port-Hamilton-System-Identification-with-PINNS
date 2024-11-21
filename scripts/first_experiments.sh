# spring
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid_sigmoid
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_matmul
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_linear_consistent
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name spring_linear_inconsistent
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid_sigmoid
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid_sigmoid_no_weight_decay --weight_decay 0
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_matmul
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_linear_consistent
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 10 --repeat 1 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid_sigmoid
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 500 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid_sigmoid --time 1
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 500 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name spring_matmul_matmul --time 1
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 500 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_linear_consistent --time 1
#
# motor
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name motor_sigmoid_sigmoid
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name motor_matmul_matmul
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_sigmoid_no_weight_decay -weight_decay 0
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name motor_sigmoid_sigmoid
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name motor_matmul_matmul
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J sigmoid  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_linear_consistent
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --epochs 500 --repeat 50 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name motor_linear_inconsistent
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name motor_sigmoid_sigmoid --time 1
python main.py --name motor --num_trajectories 100 --num_val_trajectories 1000 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name motor_matmul_matmul --time 1

# ball
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_sigmoid
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 5 --J matmul  --R matmul  --G mlp --output-weight .25 --run_name ball_matmul_matmul
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name spring_sigmoid_sigmoid_no_weight_decay --weight_decay 0
python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 5 --J linear  --R sigmoid  --G linear --output-weight .25 --run_name ball_linear_consistent
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_sigmoid
python main.py --name ball --num_trajectories 1000 --num_val_trajectories 1000 --lr 5e-3 --epochs 50 --repeat 5 --J linear  --R sigmoid  --G linear --output-weight .25 --run_name ball_linear_consistent
python main.py --name ball --num_trajectories 5000 --num_val_trajectories 1000 --lr 5e-3 --epochs 10 --repeat 5 --J sigmoid  --R sigmoid  --G mlp --output-weight .25 --run_name ball_sigmoid_sigmoid

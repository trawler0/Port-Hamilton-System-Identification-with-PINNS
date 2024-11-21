# sigmoid
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid --tag scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid --tag scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid --tag scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid --tag scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid --tag scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid --tag scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J sigmoid  --R sigmoid  --G mlp  --output-weight .25 --run_name spring_sigmoid --tag scaling

# matmul
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp  --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp  --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp  --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp  --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp  --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp  --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J matmul  --R matmul  --G mlp  --output-weight .25 --run_name spring_matmul --tag scaling

# linear
python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear  --grad_H linear --output-weight .25 --run_name spring_matmul --tag scaling
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --epochs 100 --repeat 5 --J linear  --R sigmoid  --G linear --grad_H linear --output-weight .25 --run_name spring_matmul --tag scaling
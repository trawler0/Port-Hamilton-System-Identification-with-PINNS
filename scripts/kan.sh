python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default_kan  --R default_kan --grad_H gradient_kan  --G kan --output-weight .25 --run_name ball_kan --experiment kan --hidden_dim 3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 1 --J default_kan  --R default_kan --grad_H gradient_kan  --G kan --output-weight .25 --run_name spring_kan --experiment kan --hidden_dim 3
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 1 --J default_kan  --R default_kan --grad_H gradient_kan  --G kan --output-weight .25 --run_name motor_kan --experiment kan --hidden_dim 3

python main.py --name ball --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name ball --experiment kan
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name spring --experiment kan
python main.py --name motor --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name motor --experiment kan








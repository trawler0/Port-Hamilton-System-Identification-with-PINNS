python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J linear  --R matmul  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 250 --repeat 1 --J spring  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
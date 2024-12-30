#python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 1e-3 --epochs 20000 --repeat 50 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3
#python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 1e-3 --epochs 6000 --repeat 20 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 2000 --repeat 10 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 5 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 400 --repeat 1 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J linear  --R linear  --G linear --grad_H linear --output-weight .25 --run_name wrong --experiment prior_comparison --weight_decay 1e-3

#python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 1e-3 --epochs 20000 --repeat 50 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3
#python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 1e-3 --epochs 6000 --repeat 20 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 2000 --repeat 10 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 5 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 400 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name R_generic --experiment prior_comparison --weight_decay 1e-3

#python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 1e-3 --epochs 20000 --repeat 50 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
#python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 1e-3 --epochs 6000 --repeat 20 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 2000 --repeat 10 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 5 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 400 --repeat 1 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J linear  --R quadratic  --G linear --grad_H linear --output-weight .25 --run_name R_quadratic --experiment prior_comparison --weight_decay 1e-3

#python main.py --name spring --num_trajectories 1 --num_val_trajectories 1000 --lr 1e-3 --epochs 10000 --repeat 50 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_comparison --weight_decay 1e-3
#python main.py --name spring --num_trajectories 3 --num_val_trajectories 1000 --lr 1e-3 --epochs 3000 --repeat 20 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 10 --num_val_trajectories 1000 --lr 1e-3 --epochs 1000 --repeat 10 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 30 --num_val_trajectories 1000 --lr 1e-3 --epochs 500 --repeat 5 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 100 --num_val_trajectories 1000 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 300 --num_val_trajectories 1000 --lr 1e-3 --epochs 100 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_comparison --weight_decay 1e-3
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 1e-3 --epochs 50 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_comparison --weight_decay 1e-3

python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J linear  --R default  --G linear --grad_H linear --output-weight .25 --run_name prior --experiment prior_vs_default --weight_decay 1e-3 --num_avg 1
python main.py --name spring --num_trajectories 1000 --num_val_trajectories 1000 --lr 2e-3 --epochs 100 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment prior_vs_default --num_avg 1

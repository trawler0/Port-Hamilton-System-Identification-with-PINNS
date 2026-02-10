python main.py --name ball --num_trajectories 16 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name default --experiment recipe_1
python main.py --name ball --num_trajectories 16 --num_val_trajectories 100 --lr 1e-3 --epochs 200 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name shorter --experiment recipe_1
python main.py --name ball --num_trajectories 16 --num_val_trajectories 100 --lr 1e-3 --epochs 5000 --repeat 1 --J default  --R default  --G mlp --output-weight .25 --run_name shallow --experiment recipe_1 --depth 1





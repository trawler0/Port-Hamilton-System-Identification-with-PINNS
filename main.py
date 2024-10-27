from train import TrainingModule
from model import *
from utils import Dataset, sample_initial_states, compute_metrics, visualize_trajectory, scatter
from data import simple_experiment
from pytorch_lightning import Trainer
import torch
import numpy as np
from data import dim_bias_scale
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="spring")
parser.add_argument("--num_trajectories", type=int, default=100)
parser.add_argument("--num_val_trajectories", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--J", type=str, default="matmul")
parser.add_argument("--R", type=str, default="matmul_rescale")
parser.add_argument("--G", type=str, default="mlp")
parser.add_argument("--excitation", type=str, default="linear")
parser.add_argument("--grad_H", type=str, default="gradient")
parser.add_argument("--time", type=float, default=10)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--weight_decay", type=float, default=1e-1)
parser.add_argument("--checkpoint", type=str, default="model_spring.pt")
parser.add_argument("--forecast_examples", type=int, default=5)
parser.add_argument("--forecast_time", type=float, default=100.)
parser.add_argument("--forecast_steps", type=int, default=1000)

args = parser.parse_args()

with mlflow.start_run() as run:
    time = args.time
    steps = args.steps
    dt = time / steps

    name = args.name
    DIM, scale, bias = dim_bias_scale(name)

    generator = simple_experiment(name, time, steps)
    generator_val = simple_experiment(name, time, steps)

    X0_train = sample_initial_states(args.num_trajectories, DIM,
                                     {"identifies": "uniform", "seed": args.seed, "scale": scale, "bias": bias})
    X0_val = sample_initial_states(args.num_val_trajectories, DIM,
                                   {"identifies": "uniform", "seed": args.seed, "scale": scale, "bias": bias})

    X, u, y, _ = generator.get_data(X0_train)
    X_val, u_val, y_val, trajectories_val = generator_val.get_data(X0_val)
    X, u, y = torch.tensor(X, dtype=torch.float32), torch.tensor(u, dtype=torch.float32), torch.tensor(y,
                                                                                                       dtype=torch.float32)
    X_val, u_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(u_val,
                                                                                 dtype=torch.float32), torch.tensor(
        y_val, dtype=torch.float32)

    train_ds = Dataset(X, u, y)
    val_ds = Dataset(X_val, u_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True)  # FasterLoader(X, u, y, args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    model = PHNNModel(DIM, args.hidden_dim, J=args.J, R=args.R, grad_H=args.grad_H, G=args.G,
                      excitation=args.excitation)
    model = TrainingModule(model, lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(max_epochs=args.epochs, enable_checkpointing=False, logger=False, accelerator="cpu")
    # trainer.fit(model, train_loader)
    # torch.save(model.model.state_dict(), args.checkpoint)
    model.model.load_state_dict(torch.load(args.checkpoint))

    metrics = compute_metrics(model, X_val, u_val, y_val)
    print(metrics)

    steps = args.forecast_steps
    time = args.forecast_time
    dt = time / steps
    forecast_examples = args.forecast_examples
    t = np.array([dt * s for s in range(steps)])

    generator_val = simple_experiment(name, time, steps)
    _, _, _, trajectories_val = generator_val.get_data(X0_val)
    # visualize_trajectory(model, forecast_examples, steps, dt, trajectories_val)

    X = np.concatenate([xu[0] for xu in trajectories_val])
    G_true = np.stack([np.reshape(generator_val.G(x), -1) for x in X])
    R_true = np.stack([np.reshape(generator_val.R(x), -1) for x in X])
    J_true = np.stack([np.reshape(generator_val.J(x), -1) for x in X])
    grad_H_true = np.stack([np.reshape(generator_val.grad_H(x), -1) for x in X])

    X = torch.tensor(X).float()
    J, R = model.model.reparam(X)
    G = model.model.G(X)
    grad_H = model.model.grad_H(X)

    J = np.array(J.detach())
    R = np.array(R.detach())
    G = np.array(G.detach())
    grad_H = np.array(grad_H.detach())



    if G_true.shape[1] == 1:
        G_true = np.concatenate([G_true] * G.shape[-1], -1)

    scatter(grad_H, grad_H_true, "grad_H")
    scatter(G, G_true, "G")
    scatter(R, R_true, "R")
    scatter(J, J_true, "S")



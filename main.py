from train import TrainingModule
from model import *
from utils import Dataset, sample_initial_states, compute_metrics, visualize_trajectory, scatter, get_uniform_white_noise, get_noise_bound
from data import simple_experiment
from pytorch_lightning import Trainer
import torch
import numpy as np
from data import dim_bias_scale_sigs
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="ball")
parser.add_argument("--num_trajectories", type=int, default=100)
parser.add_argument("--num_val_trajectories", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--depth", type=int, default=3)
parser.add_argument("--J", type=str, default="matmul")
parser.add_argument("--R", type=str, default="matmul")
parser.add_argument("--G", type=str, default="mlp")
parser.add_argument("--output-weight", type=float, default=.25)
parser.add_argument("--excitation", type=str, default="mlp")
parser.add_argument("--grad_H", type=str, default="gradient")
parser.add_argument("--time", type=float, default=10)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--criterion", type=str, default="normalized_mse")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--checkpoint", type=str, default="model_spring.pt")
parser.add_argument("--forecast_examples", type=int, default=20)
parser.add_argument("--forecast_time", type=float, default=100.)
parser.add_argument("--forecast_steps", type=int, default=10000)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--example", type=str, default=None)
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--dB", type=float, default=None)
parser.add_argument("--baseline", action="store_true", default=False)



args = parser.parse_args()

with mlflow.start_run(run_name=args.run_name) as run:
    mlflow.set_tag("name", args.name)
    mlflow.set_tag("training_size", args.num_trajectories)
    mlflow.set_tag("time", args.time)
    if args.tag is not None:
        mlflow.set_tag("tag", args.tag)

    mlflow.log_params(vars(args))
    mlflow.pytorch.autolog()
    time = args.time
    steps = args.steps if args.steps is not None else int(100 * args.time)
    dt = time / steps

    name = args.name
    DIM, scale, bias, sigs = dim_bias_scale_sigs(name)

    generator = simple_experiment(name, time, steps)
    generator_val = simple_experiment(name, time, steps)

    X0_train = sample_initial_states(args.num_trajectories, DIM,
                                     {"identifies": "uniform", "seed": args.seed, "scale": scale, "bias": bias})
    X0_val = sample_initial_states(args.num_val_trajectories, DIM,
                                   {"identifies": "uniform", "seed": args.seed+1, "scale": scale, "bias": bias})

    X, u, xdot, y, _ = generator.get_data(X0_train)
    X_val, u_val, xdot_val, y_val, trajectories_val = generator_val.get_data(X0_val)
    if args.dB is not None:
        a = get_noise_bound(X, args.dB)
        X = X + get_uniform_white_noise(X, a)
        X_val = X_val + get_uniform_white_noise(X_val, a)

    X, u, xdot, y = np.concatenate([X] * args.repeat), np.concatenate([u] * args.repeat), np.concatenate([xdot] * args.repeat), np.concatenate([y] * args.repeat)
    X, u, xdot, y = torch.tensor(X, dtype=torch.float32), torch.tensor(u, dtype=torch.float32), torch.tensor(xdot, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    X_val, u_val, xdot_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(u_val, dtype=torch.float32), torch.tensor(xdot_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

    train_ds = Dataset(X, u, xdot, y)
    val_ds = Dataset(X_val, u_val, xdot_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    if args.baseline:
        model = Baseline(DIM, args.hidden_dim, sigs, args.depth)
    else:
        model = PHNNModel(DIM, args.hidden_dim, args.depth, J=args.J, R=args.R, grad_H=args.grad_H, G=args.G,
                          excitation=args.excitation, u_dim=sigs)

    model = TrainingModule(model, loss_fn=args.criterion, lr=args.lr, weight_decay=args.weight_decay, output_weight=args.output_weight)
    trainer = Trainer(max_epochs=args.epochs//args.repeat, enable_checkpointing=False, logger=False, accelerator="cpu", gradient_clip_val=1)
    trainer.fit(model, train_loader)
    torch.save(model.model.state_dict(), args.checkpoint)
    model.model.load_state_dict(torch.load(args.checkpoint))

    metrics = compute_metrics(model, trajectories_val, dt, X_val, u_val, xdot_val, y_val)
    print(metrics)

    steps = args.forecast_steps
    time = args.forecast_time
    dt = time / steps
    forecast_examples = args.forecast_examples
    t = np.array([dt * s for s in range(steps)])

    generator_val = simple_experiment(name, time, steps)
    _, _, _, _, trajectories_val = generator_val.get_data(X0_val[:forecast_examples])
    if args.dB is not None:
        trajectories_val = [(x + get_uniform_white_noise(x, a), u, y, signal) for x, u, y, signal in trajectories_val]

    visualize_trajectory(model, forecast_examples, steps, dt, trajectories_val)

    if not args.baseline:
        X = np.concatenate([xu[0] for xu in trajectories_val])
        G_true = np.stack([generator_val.G(x) for x in X])
        R_true = np.stack([generator_val.R(x) for x in X])
        J_true = np.stack([generator_val.J(x) for x in X])
        grad_H_true = np.stack([generator_val.grad_H(x) for x in X])

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
        if R_true.shape[1] == 1:
            R_true = np.concatenate([R_true] * R.shape[-1], -1)
        if J_true.shape[1] == 1:
            J_true = np.concatenate([J_true] * J.shape[-1], -1)


        scatter(grad_H, grad_H_true, "grad_H")
        scatter(G, G_true, "G")
        scatter(R, R_true, "R")
        scatter(J, J_true, "S")
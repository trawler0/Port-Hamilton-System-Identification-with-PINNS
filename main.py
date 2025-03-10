from train import TrainingModule
from model import *
from utils import Dataset, sample_initial_states, compute_metrics, visualize_trajectory, scatter, \
    get_uniform_white_noise, get_noise_bound
from data import simple_experiment
from pytorch_lightning import Trainer
import torch
import numpy as np
from data import dim_bias_scale_sigs
import mlflow
import argparse
import random
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="spring")
parser.add_argument("--num_trajectories", type=int, default=100)
parser.add_argument("--num_val_trajectories", type=int, default=1000)
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--depth", type=int, default=3)
parser.add_argument("--J", type=str, default="default")
parser.add_argument("--R", type=str, default="default")
parser.add_argument("--G", type=str, default="mlp")
parser.add_argument("--output-weight", type=float, default=.25)
parser.add_argument("--excitation", type=str, default="mlp")
parser.add_argument("--grad_H", type=str, default="gradient")
parser.add_argument("--time", type=float, default=10)
parser.add_argument("--val_time", type=float, default=10)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--val_steps", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=10)
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
parser.add_argument("--affine", action="store_true", default=False)
parser.add_argument("--experiment", type=str, default="0")
parser.add_argument("--num_avg", type=int, default=1)
parser.add_argument("--rescale_epochs", type=int, default=1)
parser.add_argument("--no-forecast", action="store_true")


args = parser.parse_args()
try:
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(args.experiment)
except Exception as e:
    print(f"Error fetching or creating experiment: {e}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(args.seed)
torch.use_deterministic_algorithms(True)

with mlflow.start_run(run_name=args.run_name, experiment_id=experiment_id):
    mlflow.set_tag("name", args.name)
    mlflow.set_tag("training_size", args.num_trajectories)
    mlflow.set_tag("time", args.time)
    if args.tag is not None:
        mlflow.set_tag("tag", args.tag)

    mlflow.log_params(vars(args))
    mlflow.pytorch.autolog()
    time = args.time
    steps = args.steps if args.steps is not None else int(100 * args.time)
    val_steps = args.val_steps if args.val_steps is not None else int(100 * args.val_time)
    dt = time / steps

    name = args.name
    DIM, scale, bias, sigs, amplitude_train, f0_train, amplitude_val, f0_val = dim_bias_scale_sigs(name)

    generator = simple_experiment(name, time, steps, amplitude_train, f0_train)
    generator_val = simple_experiment(name, args.val_time, val_steps, amplitude_val, f0_val)

    X0_train = sample_initial_states(args.num_trajectories, DIM,
                                     {"identifies": "uniform", "seed": args.seed, "scale": scale, "bias": bias})
    X0_val = sample_initial_states(args.num_val_trajectories, DIM,
                                   {"identifies": "uniform", "seed": args.seed + 1, "scale": scale, "bias": bias})

    X, u, xdot, y, _ = generator.get_data(X0_train)
    X_val, u_val, xdot_val, y_val, trajectories_val = generator_val.get_data(X0_val)

    scaler_X = StandardScaler()
    scaler_u = StandardScaler()
    X = scaler_X.fit_transform(X)
    u = scaler_u.fit_transform(u)

    class Predictor(nn.Module):

        def __init__(self, model, scaler_X, scaler_u):
            super().__init__()
            self.model = model
            self.scaler_X = scaler_X
            self.scaler_u = scaler_u

        def forward(self, x, u):
            x = x.detach().numpy()
            u = u.detach().numpy()
            x = self.scaler_X.transform(x)
            u = self.scaler_u.transform(u)
            x = torch.tensor(x).float()
            u = torch.tensor(u).float()
            xdot, y = self.model(x, u)
            return xdot, y

    if args.dB is not None:
        a = get_noise_bound(u, args.dB)
        b = get_noise_bound(y, args.dB)
        u = u + get_uniform_white_noise(u, a)
        y = y + get_uniform_white_noise(y, b)
        traj_val = []
        for x_, u_, y_, signal in trajectories_val:
            u_ = u_ + get_uniform_white_noise(u_, a)
            y_ = y_ + get_uniform_white_noise(y_, b)
            traj_val.append((x_, u_, y_, signal))
        trajectories_val = traj_val
    else:
        a = None
        b = None

    X, u, xdot, y = np.concatenate([X] * args.repeat), np.concatenate([u] * args.repeat), np.concatenate(
        [xdot] * args.repeat), np.concatenate([y] * args.repeat)
    X, u, xdot, y = torch.tensor(X, dtype=torch.float32), torch.tensor(u, dtype=torch.float32), torch.tensor(xdot,
                                                                                                             dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32)
    X_val, u_val, xdot_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(u_val,
                                                                                           dtype=torch.float32), torch.tensor(
        xdot_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

    train_ds = Dataset(X, u, xdot, y)
    val_ds = Dataset(X_val, u_val, xdot_val, y_val)

    models = []
    for i in range(args.num_avg):
        set_seed(args.seed + i)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        assert not (args.baseline and args.affine)
        if args.baseline:
            model = Baseline(DIM, 2 * args.hidden_dim, sigs, args.depth)
        elif args.affine:
            model = InputAffine(DIM, 2 * args.hidden_dim, sigs, args.depth)
        else:
            model = PHNNModel(DIM, args.hidden_dim, args.depth, J=args.J, R=args.R, grad_H=args.grad_H, G=args.G,
                              excitation=args.excitation, u_dim=sigs)

        model = TrainingModule(model, loss_fn=args.criterion, lr=args.lr, weight_decay=args.weight_decay,
                               output_weight=args.output_weight)
        trainer = Trainer(max_epochs=int(args.epochs // args.repeat * args.rescale_epochs), enable_checkpointing=False, logger=False,
                          accelerator="cpu", gradient_clip_val=1)
        trainer.fit(model, train_loader)
        model = Predictor(model.model, scaler_X, scaler_u) # for eval
        models.append(model)
    mlflow.pytorch.log_model(model, "model")

    metrics = compute_metrics(models, trajectories_val, dt, X_val, u_val, xdot_val, y_val)
    print(metrics)

    if not args.no_forecast:
        steps = args.forecast_steps
        time = args.forecast_time
        dt = time / steps
        forecast_examples = args.forecast_examples
        t = np.array([dt * s for s in range(steps)])

        generator_val = simple_experiment(name, time, steps, amplitude_val, f0_val)
        _, _, _, _, trajectories_val = generator_val.get_data(X0_val[:forecast_examples])
        visualize_trajectory(model, forecast_examples, steps, dt, trajectories_val, a=None, b=None)

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

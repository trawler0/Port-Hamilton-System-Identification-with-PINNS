import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import math
import mlflow
from io import BytesIO
from matplotlib import pyplot as plt
import tempfile
import os


def sample_initial_states(num_trajectories, dim, strategy):
    np.random.seed(strategy["seed"])
    if strategy["identifies"] == "uniform":
        X0 = np.random.uniform(0, 1, (num_trajectories, dim))
    elif strategy["identifies"] == "normal":
        X0 = np.random.normal(0, 1, (num_trajectories, dim))
    else:
        raise NotImplementedError("Sampling strategy not implemented")
    if "scale" in strategy:
        X0 = X0 * np.reshape(strategy["scale"], (1, dim))
    if "bias" in strategy:
        X0 = X0 + np.reshape(strategy["bias"], (1, dim))
    return X0

def multi_sin_signal(n_signals=1, amplitude=.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    i = np.arange(5)
    i, phi = np.stack([i] * n_signals), np.random.uniform(0, 2 * np.pi, (n_signals, 5))
    def u(t):
        sins = np.sin(2 * np.pi * i * .1 * t / 10 + phi)
        out = np.sum(sins, axis=-1) * amplitude
        return out
    return u


@torch.no_grad()
def forecast(model, X0, u, dt, signal, steps, clamp=100., a=None, b=None):
    model.eval()
    X = torch.zeros(X0.shape[0], steps, X0.shape[-1])
    u = torch.tensor(u)
    X[:, 0] = torch.tensor(X0)
    t = 0
    for i in tqdm(range(0, steps-1)):
        X = X.clone().detach()
        """# first order approximation
        with torch.autograd.enable_grad():
            dxdt = model(X[:, i].float(), u[:, i].float())[0]
        X[:, i+1] = dxdt * dt + X[:, i]"""
        # runge kutta
        u0 = torch.stack([torch.tensor(sig(t)) for sig in signal]).float()
        u_mid = torch.stack([torch.tensor(sig(t + dt/2)) for sig in signal]).float()
        u_end = torch.stack([torch.tensor(sig(t + dt)) for sig in signal]).float()
        if a is not None:
            u0 = torch.tensor(u[:, i]).float()
            u_mid += torch.tensor(get_uniform_white_noise(u_mid, a)).float()
            u_end += torch.tensor(u[:, i+1]).float()
        k1 = model(X[:, i], u0)[0]
        k2 = model(X[:, i] + k1 * dt / 2, u_mid)[0]
        k3 = model(X[:, i] + k2 * dt / 2, u_mid)[0]
        k4 = model(X[:, i] + k3 * dt, u_end)[0]
        X[:, i+1] = X[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        X[:, i+1] = torch.clamp(X[:, i+1], -clamp, clamp)
        t += dt
    return X


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, u, xdot, y):
        self.X = X
        self.u = u
        self.xdot = xdot
        self.y = y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.u[idx], self.xdot[idx], self.y[idx]


def normalized_mse(x, target):
    std = target.std()
    return ((x - target) / (std + 1e-6)).pow(2).mean()

def normalized_mae(x, target):
    std = target.std()
    return ((x - target) / (std + 1e-6)).abs().mean()

@torch.no_grad()
def compute_metrics(model, trajectories, dt, X, u, xdot, y):

    if isinstance(model, list):
        out_dict = {}
        all_metrics = {}
        for m in model:
            metrics = compute_metrics(m, trajectories, dt, X, u, xdot, y)
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
        for k, v in all_metrics.items():
            out_dict[k] = np.mean(v)
        return out_dict
    else:
        N = len(X)
        x_dot_pred = []
        y_pred = []
        for i in tqdm(range(N // 512 + 1)):
            with torch.enable_grad():
                out = model(X[i*512:(i+1)*512], u[i*512:(i+1)*512])
            x_dot_pred.append(out[0].detach())
            y_pred.append(out[1].detach())
        xdot_pred = torch.cat(x_dot_pred, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        mae = torch.abs(xdot_pred - xdot).mean().item()
        mse = ((xdot_pred - xdot)**2).mean().item()
        mae_rel = normalized_mae(xdot_pred, xdot).item()
        mse_rel = normalized_mse(xdot_pred, xdot).item()

        output_mae = torch.abs(y_pred - y).mean().item()
        output_mse = ((y_pred - y)**2).mean().item()
        output_mae_rel = normalized_mae(y_pred, y).item()
        output_mse_rel = normalized_mse(y_pred, y).item()

        out_dict = {
            "mae": mae,
            "mse": mse,
            "mae_rel": mae_rel,
            "mse_rel": mse_rel,
            "output_mae": output_mae,
            "output_mse": output_mse,
            "output_mae_rel": output_mae_rel,
            "output_mse_rel": output_mse_rel
        }
        """X, u, y, signal = np.stack([t[0] for t in trajectories]), np.stack([t[1] for t in trajectories]), np.stack(
            [t[2] for t in trajectories]), [t[3] for t in trajectories]
        X0 = X[:, 0]
        X_pred = forecast(model, X0, u, dt, signal, X.shape[1]).detach().numpy()
        forecast_mae_rel = normalized_mae(torch.tensor(X_pred), torch.tensor(X)).item()
        forecast_mse_rel = normalized_mse(torch.tensor(X_pred), torch.tensor(X)).item()
        forecast_mae = torch.abs(torch.tensor(X_pred) - torch.tensor(X)).mean().item()
        forecast_mse = ((torch.tensor(X_pred) - torch.tensor(X))**2).mean().item()
        out_dict["forecast_mae"] = forecast_mae
        out_dict["forecast_mse"] = forecast_mse
        out_dict["forecast_mae_rel"] = forecast_mae_rel
        out_dict["forecast_mse_rel"] = forecast_mse_rel"""

        if mlflow.active_run() is not None:
            mlflow.log_metrics(out_dict)

        return out_dict


def visualize_trajectory(model, forecast_examples, steps, dt, trajectories, a=None, b=None):

    t = np.array([dt * s for s in range(steps)])

    X, u, y, signal = np.stack([t[0] for t in trajectories]), np.stack([t[1] for t in trajectories]), np.stack([t[2] for t in trajectories]), [t[3] for t in trajectories]
    X0 = X[:, 0]
    X_pred = forecast(model, X0, u, dt, signal, steps, a=a, b=b).detach().numpy()
    # log predictions as numpy
    tmpdir = tempfile.TemporaryDirectory()
    np.save(os.path.join(tmpdir.name, "X_pred.npy"), X_pred)
    np.save(os.path.join(tmpdir.name, "X.npy"), X)
    if mlflow.active_run() is not None:
        mlflow.log_artifact(os.path.join(tmpdir.name, "X_pred.npy"))
        mlflow.log_artifact(os.path.join(tmpdir.name, "X.npy"))
    for j in range(forecast_examples):
        torch.cuda.empty_cache()

        fig, axs = plt.subplots(X.shape[-1] + u.shape[-1], 1, figsize=(30, 10))
        fig.suptitle(f'Trajectory example', fontsize=16)

        for i in range(X.shape[-1]):
            axs[i].plot(t, X[j, :, i], linestyle="dashed", color="red", label=f"X_{i}", linewidth=4)
            axs[i].plot(t, X_pred[j, :, i], linestyle="dotted", color="blue", label=f"X_pred_{i}", linewidth=4)
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel(f"X_{i}")
            axs[i].set_title(f"Trajectory example {j + 1}")
            axs[i].grid(True)
        for k in range(u.shape[-1]):
            col = ["green", "orange", "purple", "black"]
            axs[X.shape[-1] + k].plot(t, u[j, :, k], linestyle="solid", color=col[k], label=f"u_{k}", linewidth=2)
            axs[X.shape[-1] + k].set_xlabel("Time")
            axs[X.shape[-1] + k].set_ylabel(f"u_{k}")
            axs[X.shape[-1] + k].set_title(f"Trajectory example {j + 1}")
            axs[X.shape[-1] + k].grid(True)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer to the beginning

        # Log the figure to MLflow
        if mlflow.active_run() is not None:
            mlflow.log_figure(fig, f"trajectory_example_{j + 1}.png")

        # Close the figure to free up memory
        plt.close(fig)


def scatter(cp, c, name, samples=1000):

    cp = np.reshape(cp, (cp.shape[0], -1))
    c = np.reshape(c, (c.shape[0], -1))


    assert c.shape == cp.shape

    idx = list(range(len(c)))
    np.random.shuffle(idx)

    c = c[idx[:samples]]
    cp = cp[idx[:samples]]
    for j in range(c.shape[-1]):

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'{name}', fontsize=16)
        axs.scatter(c[:, j], cp[:, j])
        axs.set_xlabel(f"True {name}_{j}")
        axs.set_ylabel(f"Predicted {name}_{j}")
        axs.set_title(f"Scatter plot of {name}_{j}")
        axs.grid(True)

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer to the beginning

        # Log the figure to MLflow
        if mlflow.active_run() is not None:
            mlflow.log_figure(fig, f"scatter_{name}_{j}.png")

        # Close the figure to free up memory
        plt.close(fig)

def get_noise_bound(X, sig_noise_ratio):
    power_signal = np.mean(X ** 2, axis=0)  # N x S
    # SNR (db): 10 * log10(P_signal / P_noise)
    P_noise = 10 ** (-sig_noise_ratio / 10) * power_signal
    # P_noise**2 = (2 a)**2 / 12
    a = np.sqrt(3 * P_noise)  # S
    #noise = np.random.uniform(-a, a, X.shape)
    #X += noise
    return a

def get_uniform_white_noise(X, a):
    noise = np.random.uniform(-a, a, X.shape)
    return noise



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    u = multi_sin_signal(n_signals=2, amplitude=.2)
    t = np.linspace(0, 10, 1000)
    print(u(0))
    plt.plot(t, [u(tt)[0] for tt in t])
    plt.plot(t, [u(tt)[1] for tt in t])
    plt.show()
    print(np.mean(u, axis=0))
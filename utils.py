import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import math
import mlflow
from io import BytesIO
from matplotlib import pyplot as plt


def sample_initial_states(num_trajectories, dim, strategy):
    if strategy["identifies"] == "uniform":
        np.random.seed(strategy["seed"])
        X0 = np.random.uniform(-1, 1, (num_trajectories, dim))
    elif strategy["identifies"] == "normal":
        np.random.seed(strategy["seed"])
        X0 = np.random.normal(0, 1, (num_trajectories, dim))
    else:
        raise NotImplementedError("Sampling strategy not implemented")
    if "scale" in strategy:
        X0 = X0 * strategy["scale"]
    if "bias" in strategy:
        X0 = X0 + strategy["bias"]
    return X0

def generate_signal(period_min, period_max, amplitude_min, amplitude_max, bias_min, bias_max, seed, signal, n_signals=1):
    np.random.seed(seed)
    period = np.random.uniform(period_min, period_max)
    amplitude = np.random.uniform(amplitude_min, amplitude_max)
    bias = np.random.uniform(bias_min, bias_max)
    def u(t):
        if signal == "sin":
            return np.reshape(np.sin(2 * np.pi * t / period + bias + bias) * amplitude, (1, ))
        elif signal == "sin_pos":
            return np.reshape((np.sin(2 * np.pi * t / period + bias + bias) + 1)*amplitude/2, (1, ))

        elif signal == "binary":
            return np.reshape(amplitude if t % period < period / 2 else 0, (1, ))
        else:
            raise NotImplementedError("Signal not implemented")

    if n_signals > 1:
        def u_cat(t):
            u_ = [generate_signal(period_min, period_max, amplitude_min, amplitude_max, bias_min, bias_max, seed + j, signal) for j in range(n_signals)]
            return np.concatenate([u(t) for u in u_])
        return u_cat
    else:
        return u

def forecast(model, X0, u, dt, steps, clamp=10.):
    X = torch.zeros(steps, X0.shape[0])
    u = torch.tensor(u)
    X[0] = torch.tensor(X0)
    for i in range(1, steps):
        X = X.clone().detach()
        X[i] = model(X[i-1:i].float(), u[i-1:i].float())[0] * dt + X[i-1:i]
        X[i] = torch.clamp(X[i], -clamp, clamp)
    return X


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, u, xdot, y):
        print(X.shape, u.shape, xdot.shape, y.shape)
        self.X = X
        self.u = u
        self.xdot = xdot
        self.y = y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.u[idx], self.xdot[idx], self.y[idx]


class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, trajectories):
        self.trajectories = trajectories


    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]



def normalized_mse(x, target):
    std = target.std()
    return ((x - target) / (std + 1e-6)).pow(2).mean()

def normalized_mae(x, target):
    std = target.std()
    return ((x - target) / (std + 1e-6)).abs().mean()



# this stuff is not working properly, but it would accelerate training
# it is disposed to later

class FasterLoader:

    def __init__(self, X, u, y, batch_size):
        self.X = X
        self.u = u
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        X, u, y = deepcopy(self.X), deepcopy(self.u), deepcopy(self.y)
        idx = torch.randperm(len(X))
        for i in range(len(X) // self.batch_size):
            x_out = X[idx[i * self.batch_size: (i + 1) * self.batch_size]]
            u_out = u[idx[i * self.batch_size: (i + 1) * self.batch_size]]
            y_out = y[idx[i * self.batch_size: (i + 1) * self.batch_size]]
            yield x_out, u_out, y_out

    def __len__(self):
        return len(self.X) // self.batch_size


# disposed to later
def train(model, X, u, y, epochs, lr, weight_decay, batch_size, criterion=normalized_mse):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = FasterLoader(X, u, y, batch_size)
    total_steps = len(X) // batch_size * epochs
    def lr_lambda(step, warmup=.1):
        if step / total_steps < warmup:
            return step / (total_steps * warmup)
        actual_step = step - total_steps * warmup
        actual_total = total_steps * (1 - warmup)
        return .5 * (1 + math.cos(math.pi * actual_step / actual_total))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for epoch in range(epochs):
        pbar = tqdm(loader)
        model.train()
        running_loss = 0
        for i, (x, u, y) in enumerate(pbar):
            optimizer.zero_grad()
            y_hat = model(x, u)
            loss = criterion(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": running_loss / (i + 1), "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})

    return model


def compute_metrics(model, X_val, u_val, xdot_val, y_val):

    xdot_pred, y_pred = model(X_val, u_val)
    mae = torch.abs(xdot_pred - xdot_val).mean().item()
    mse = ((xdot_pred - xdot_val)**2).mean().item()
    mae_rel = normalized_mae(xdot_pred, xdot_val).item()
    mse_rel = normalized_mse(xdot_pred, xdot_val).item()

    output_mae = torch.abs(y_pred - y_val).mean().item()
    output_mse = ((y_pred - y_val)**2).mean().item()
    output_mae_rel = normalized_mae(y_pred, y_val).item()
    output_mse_rel = normalized_mse(y_pred, y_val).item()

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
    if mlflow.active_run() is not None:
        mlflow.log_metrics(out_dict)

    return out_dict


def visualize_trajectory(model, forecast_examples, steps, dt, trajectories):

    t = np.array([dt * s for s in range(steps)])

    for j in range(forecast_examples):
        X, u, y = trajectories[j]
        X0 = X[0]
        X_pred = forecast(model, X0, u, dt, steps).detach().numpy()
        torch.cuda.empty_cache()

        fig, axs = plt.subplots(X.shape[1], 1, figsize=(30, 10))
        fig.suptitle(f'Trajectory example', fontsize=16)

        for i in range(X.shape[1]):
            axs[i].plot(t, X[:, i], linestyle="dashed", color="red", label=f"X_{i}", linewidth=4)
            axs[i].plot(t, X_pred[:, i], linestyle="dotted", color="blue", label=f"X_pred_{i}", linewidth=4)
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

        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer to the beginning

        # Log the figure to MLflow
        if mlflow.active_run() is not None:
            mlflow.log_figure(fig, f"scatter_{name}_{j}.png")

        # Close the figure to free up memory
        plt.close(fig)









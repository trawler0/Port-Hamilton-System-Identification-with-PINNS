import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import math


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

def generate_signal(period_min, period_max, amplitude_min, amplitude_max, seed, signal):
    np.random.seed(seed)
    period = np.random.uniform(period_min, period_max)
    amplitude = np.random.uniform(amplitude_min, amplitude_max)
    def u(t):
        if signal == "sin":
            return np.sin(2 * np.pi * t / period) * amplitude
        elif signal == "binary":
            return amplitude if t % period < period / 2 else 0
        else:
            raise NotImplementedError("Signal not implemented")
    return u

def forecast(model, X0, u, dt, steps):
    X = torch.zeros((len(X0), steps, X0.shape[1]))
    X[:, 0] = torch.tensor(X0)
    for i in range(1, steps):
        X = X.clone().detach()
        X[:, i] = model(X[:, i - 1], u) * dt + X[:, i - 1]
    return X


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, u, y):
        self.X = X
        self.u = u
        self.y = y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.u[idx], self.y[idx]


class ValDataset(torch.utils.data.Dataset):

    def __init__(self, trajectories):
        self.trajectories = trajectories


    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


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


def normalized_mse(x, target):
    std = target.std()
    return ((x - target) / std).pow(2).mean()

def normalized_mae(x, target):
    std = target.std()
    return ((x - target) / std).abs().mean()



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



if __name__ == "__main__":
    from model import PHNNModel
    model = PHNNModel(4, 64, J="sigmoid", R="sigmoid", grad_H="gradient")
    model.load_state_dict(torch.load('model_spring.pt'))







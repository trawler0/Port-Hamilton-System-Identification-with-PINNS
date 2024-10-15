import numpy as np
import torch


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


class TrainDataset(torch.utils.data.Dataset):

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


def normalized_mse(x, target):
    std = target.std()
    return ((x - target) / std).pow(2).mean()

def normalized_mae(x, target):
    std = target.std()
    return ((x - target) / std).abs().mean()


if __name__ == "__main__":
    x = torch.rand(10, 4, 2)
    target = torch.rand(10, 4, 2)
    print(normalized_mse(x, target))
import numpy as np
from scipy.integrate import odeint
import torch
from utils import sample_initial_states, generate_signal
from functools import partial
from tqdm import tqdm


class BaseDataGenerator:
    X: torch.Tensor
    y: torch.Tensor

    def __init__(self, simulation_time, num_steps, generate_signal=None):
        self.simulation_time = simulation_time
        self.num_steps = num_steps
        self.generate_signal = generate_signal

    def H(self, x):
        raise NotImplementedError("Subclass must implement abstract method")

    def grad_H(self, x):
        raise NotImplementedError("Subclass must implement abstract method")

    def J(self, x):
        raise NotImplementedError("Subclass must implement abstract method")

    def R(self, x):
        raise NotImplementedError("Subclass must implement abstract method")

    def G(self, x):
        raise NotImplementedError("Subclass must implement abstract method")

    def system_dynamics(self, x, t, u):
        if u is None:
            u = lambda s: 0
        return (self.J(x) - self.R(x)) @ self.grad_H(x) + self.G(x) * u(t)

    def __call__(self, x, u):
        y_hat = []
        for i in range(len(x)):
            y_hat.append((self.J(x[i]) - self.R(x[i])) @ self.grad_H(x[i]) + self.G(x[i]) * u[i])
        return np.stack(y_hat)

    def generate_trajectory(self, x0, u=None):
        t = np.linspace(0, self.simulation_time, self.num_steps)
        dynamics = partial(self.system_dynamics, u=u)
        if u is None:
            u_out = np.zeros((self.num_steps, 1))
        else:
            u_out = np.array([u(t) for t in t]).reshape(-1, 1)
        return odeint(dynamics, x0, t), u_out

    def get_data(self, X0):
        all_X = []
        all_u = []
        all_y = []
        print("Generating trajectories")
        dt = self.simulation_time / self.num_steps
        trajectories = []
        for j, x0 in enumerate(tqdm(X0)):
            u = self.generate_signal(seed=j) if self.generate_signal is not None else None
            X, u = self.generate_trajectory(x0, u)
            y = (X[1:] - X[:-1]) / dt
            trajectories.append((X, u))
            all_X.append(X[:-1])
            all_u.append(u[:-1])
            all_y.append(y)

        X = np.concatenate(all_X)
        u = np.concatenate(all_u)
        y = np.concatenate(all_y)
        return X, u, y, trajectories


class CoupledSpringMassDamper(BaseDataGenerator):

    def __init__(self, masses, spring_constants, damping, simulation_time, num_steps, generate_signal=None):
        super().__init__(simulation_time, num_steps, generate_signal)
        self.m1, self.m2 = masses
        self.k1, self.k2 = spring_constants
        self.damping = damping
        self.M = np.diag([spring_constants[0] / 2, 1 / (2 * masses[0]), spring_constants[1] / 2, 1 / (2 * masses[1])])

    def H(self, x):
        return x.T @ self.M @ x

    def grad_H(self, x):
        return self.M @ x  # error? should be 2Mx

    def J(self, x):
        return np.array([[0, 1, 0, 0],
                         [-1, 0, 1, 0],
                         [0, -1, 0, 1],
                         [0, 0, -1, 0]])

    def R(self, x):
        R = np.array([[0, 0, 0, 0],
                      [0, self.damping * x[1] ** 2 / self.m1 ** 2, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, self.damping * x[3] ** 2 / self.m2 ** 2]])
        return R

    def G(self, x):
        return 0

    def u(self, t, *args):
        return np.zeros((1,))


class ControlledInductor(BaseDataGenerator):

    def __init__(self, m, R, c, G, simulation_time, num_steps, generate_signal=None):
        super().__init__(simulation_time, num_steps, generate_signal)
        self.m = m
        self.R_ = R
        self.c = c
        self.G_ = G

    def L(self, x1):
        return 1 / (0.1 + x1 ** 2)

    def H(self, x):
        return 0.5 * x[1] ** 2 / self.m + 0.5 * x[2] ** 2 / self.L(x[0])

    def grad_H(self, x):
        dH_dx1 = 0.5 * x[2] ** 2 * (2 * x[0]) / (0.1 + x[0] ** 2) ** 2
        dH_dx2 = x[1] / self.m
        dH_dx3 = x[2] / self.L(x[0])
        return np.array([dH_dx1, dH_dx2, dH_dx3])

    def J(self, x):
        return np.array([[0, 1, 0],
                         [-1, -self.c * np.abs(x[1]), 0],
                         [0, 0, -1 / self.R_]])

    def R(self, x):
        return 0

    def u(self, t, b=0.):
        period = 5.
        amplitude = 1.
        u = np.sin(2 * np.pi * t / period + b) * amplitude
        return np.ones((1,)) * u

    def G(self, x):
        return self.G_


def simple_experiment(name, simulation_time, num_steps, **kwargs):
    if name == "spring":
        masses = kwargs.pop("masses", (1., 1.5))
        spring_constants = kwargs.pop("spring_constants", (1., .1))
        damping = kwargs.pop("damping", 2.)
        return CoupledSpringMassDamper(masses, spring_constants, damping, simulation_time, num_steps)
    elif name == "inductor":
        m = kwargs.pop("m", 1)
        R = kwargs.pop("R", .1)
        c = kwargs.pop("c", 1)
        G = kwargs.pop("G", np.array([0, 0, 1]))
        amplitude_min = kwargs.pop("amplitude_min", 0)
        amplitude_max = kwargs.pop("amplitude_max", 1)
        period_min = kwargs.pop("period_min", .5)
        period_max = kwargs.pop("period_max", 10)
        signal = kwargs.pop("signal", "sin")
        get_u = partial(generate_signal, period_min, period_max, amplitude_min, amplitude_max, signal=signal)

        return ControlledInductor(m, R, c, G, simulation_time, num_steps, get_u)
    else:
        raise NotImplementedError("Experiment not implemented")

def dim_bias_scale(name):
    if name == "spring":
        DIM = 4
        scale = np.array([1., 1., 1., 1.])
        bias = np.array([0., 0., 0., 0.])
    elif name == "inductor":
        DIM = 3
        scale = np.array([[1.5, .4, 1.]])
        bias = np.array([[-.75, -.2, .3]])
    else:
        raise ValueError("Unknown problem")
    return DIM, scale, bias

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    generator = simple_experiment("inductor", 10, 1000)
    X0 = sample_initial_states(100, 3, {"identifies": "uniform", "seed": 42})
    X, u, y, trajectories = generator.get_data(X0)

    y_hat = generator(X, u)
    std = np.std(y, axis=0, keepdims=True)
    print(std)
    print(np.mean(np.abs(y_hat - y) / std), np.mean(np.abs(y_hat - y)))

    X, u = trajectories[5]
    y_hat = generator(X[:-1], u[:-1])
    dt = 1 / 100
    y = (X[1:] - X[:-1]) / dt
    X = X[:-1]
    u = u[:-1]

    plt.plot(X[:, 0], label="X")
    plt.plot(y_hat[:, 0], label="y_hat")
    plt.plot(y[:, 0], label="y")
    plt.plot(u, label="u")
    plt.legend()
    plt.show()

"""
x0
x1_pred = x0 + f(x0)
x2_pred = x1_pred + f(x1)
x3_pred = x2_pred + f(x2)
x4_pred = x3_pred + f(x3)

"""

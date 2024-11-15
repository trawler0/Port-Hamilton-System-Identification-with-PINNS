import numpy as np
from scipy.integrate import odeint
import torch
from utils import sample_initial_states, multi_sin_signal
from functools import partial
from tqdm import tqdm


class BaseDataGenerator:

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

    def y(self, x):
        grad_H = self.grad_H(x)
        G = self.G(x)
        return grad_H.dot(G)

    def system_dynamics(self, x, t, u):
        if u is None:
            u = lambda s: np.zeros((self.G(x).shape[-1], ))
        return (self.J(x) - self.R(x)) @ self.grad_H(x) + self.G(x).dot(u(t))

    def __call__(self, x, u):
        xdot_hat = []
        for i in range(len(x)):
            xdot_hat.append((self.J(x[i]) - self.R(x[i])) @ self.grad_H(x[i]) + self.G(x[i]).dot(u[i]))
        return np.stack(xdot_hat)

    def generate_trajectory(self, x0, u=None):
        t = np.linspace(0, self.simulation_time, self.num_steps)
        dynamics = partial(self.system_dynamics, u=u)
        if u is None:
            u_out = np.zeros((self.num_steps, 1))
        else:
            u_out = np.stack([u(t) for t in t])
        return odeint(dynamics, x0, t), u_out

    def get_data(self, X0):
        all_X = []
        all_u = []
        all_xdot = []
        all_y = []
        print("Generating trajectories")
        dt = self.simulation_time / self.num_steps
        trajectories = []
        for j, x0 in enumerate(tqdm(X0)):
            u = self.generate_signal(seed=j) if self.generate_signal is not None else None
            X, u = self.generate_trajectory(x0, u)
            xdot = (X[1:] - X[:-1]) / dt
            # xdot = self.__call__(X[:-1], u[:-1])
            y = np.stack([self.y(x) for x in X])
            trajectories.append((X, u, y))
            all_X.append(X[:-1])
            all_u.append(u[:-1])
            all_xdot.append(xdot)
            all_y.append(y[:-1])

        X = np.concatenate(all_X)
        u = np.concatenate(all_u)
        xdot = np.concatenate(all_xdot)
        y = np.concatenate(all_y)
        return X, u, xdot, y, trajectories


class CoupledSpringMassDamper(BaseDataGenerator):

    def __init__(self, G_, masses, spring_constants, damping, simulation_time, num_steps, generate_signal=None):
        super().__init__(simulation_time, num_steps, generate_signal)
        self.G_ = G_
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
        return self.G_

    def u(self, t, *args):
        return np.zeros((1,))


class MagneticBall(BaseDataGenerator):

    def __init__(self, m, R, c, G, simulation_time, num_steps, generate_signal=None):
        super().__init__(simulation_time, num_steps, generate_signal)
        self.m = m
        self.R_ = R
        self.c = c
        self.G_ = G

    def L(self, x1):
        return 1 / (0.1 + x1 ** 2)

    def H(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return .5 * (x2 ** 2 / self.m + x3 ** 2 / self.L(x1))

    def grad_H(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        dH_dx1 = x3**2 * x1
        dH_dx2 = x2 / self.m
        dH_dx3 = x3 / self.L(x1)
        return np.array([dH_dx1, dH_dx2, dH_dx3])

    def J(self, x):
        return np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 0]])

    def R(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return np.array([[0, 0, 0],
                         [0, self.c * np.abs(x2), 0],
                         [0, 0, 1 / self.R_]])

    def G(self, x):
        return self.G_

class PMSM(BaseDataGenerator):

    def __init__(self, J_m, L, beta, r, Phi, G, simulation_time, num_steps, generate_signal=None):
        super().__init__(simulation_time, num_steps, generate_signal)
        self.J_m = J_m
        self.L = L
        self.beta = beta
        self.r = r
        self.Phi = Phi
        self.G_ = G


    def H(self, x):
        phi_d = x[0]
        phi_q = x[1]
        p = x[2]
        return .5 * (phi_d**2 / self.L + phi_q**2 / self.L + p**2 / self.J_m)

    def grad_H(self, x):
        phi_d = x[0]
        phi_q = x[1]
        p = x[2]
        return np.array([phi_d / self.L, phi_q / self.L, p / self.J_m])


    def J(self, x):
        phi_d = x[0]
        phi_q = x[1]
        p = x[2]
        return np.array([[0, 0, phi_q],
                         [0, 0, -phi_d - self.Phi],
                         [-phi_q, phi_d + self.Phi, 0]])

    def R(self, x):
        phi_d = x[0]
        phi_q = x[1]
        p = x[2]
        return np.array([[self.r, 0, 0],
                        [0, self.r, 0],
                        [0, 0, self.beta]])

    def G(self, x):
        return self.G_


def simple_experiment(name, simulation_time, num_steps, **kwargs):
    if name == "spring":
        G = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
        masses = kwargs.pop("masses", (1., 1.5))
        spring_constants = kwargs.pop("spring_constants", (1., .1))
        damping = kwargs.pop("damping", 2.)
        get_u = partial(multi_sin_signal, n_signals=2, amplitude=kwargs.pop("amplitude", .2))
        return CoupledSpringMassDamper(G, masses, spring_constants, damping, simulation_time, num_steps, get_u)
    elif name == "ball":
        m = kwargs.pop("m", 1.)  # Hannes: 0.012, Achraf: 1.
        R = kwargs.pop("R", .1)  # Hannes: 0.1, Achraf: 0.1
        c = kwargs.pop("c", 1.)  # Hannes: 0.1, Achraf: 1.
        G = kwargs.pop("G", np.array([[0], [0], [1]]))
        get_u = partial(multi_sin_signal, amplitude=kwargs.pop("amplitude", .2))
        return MagneticBall(m, R, c, G, simulation_time, num_steps, get_u)
    elif name == "motor":
        J_m = kwargs.pop("J_m", 1)
        L = kwargs.pop("L", 1)
        beta = kwargs.pop("beta", 1)
        r = kwargs.pop("r", 1)
        Phi = kwargs.pop("Phi", 1)
        G = kwargs.pop("G", np.array([[1, 0], [0, 1], [0, 0]]))
        get_u = partial(multi_sin_signal, n_signals=2, amplitude=kwargs.pop("amplitude", .2))
        return PMSM(J_m, L, beta, r, Phi, G, simulation_time, num_steps, get_u)
    else:
        raise NotImplementedError("Experiment not implemented")

def dim_bias_scale_sigs(name):
    if name == "spring":
        DIM = 4
        scale = np.array([1., 1., 1., 1.])
        bias = np.array([0., 0., 0., 0.])
        sigs = 2
    elif name == "ball":
        DIM = 3
        scale = np.array([[1.5, .4, 1.]])
        bias = np.array([[-.75, -.2, .3]])
        sigs = 1
    elif name == "motor":
        DIM = 3
        scale = np.array([[1., 1., 1.]])
        bias = np.array([[0., 0., 0.]])
        sigs = 2
    else:
        raise ValueError("Unknown problem")
    return DIM, scale, bias, sigs

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    generator = simple_experiment("spring", 10, 1000)
    X0 = sample_initial_states(20, 4, {"identifies": "uniform", "seed": 41})
    X, u, xdot, y, trajectories = generator.get_data(X0)
    print(y.shape)

    xdot_hat = generator(X, u)
    print(xdot.shape, xdot_hat.shape)
    std = np.std(xdot, axis=0, keepdims=True)
    print(std)
    print(np.mean(np.abs(xdot_hat - xdot) / std), np.mean(np.abs(xdot_hat - xdot)))

    for j in range(10):
        X, u, y = trajectories[j]
        dt = 1 / 100
        xdot = (X[1:] - X[:-1]) / dt
        X = X[:-1]
        u = u[:-1]

        for k in range(X.shape[-1]):
            plt.plot(X[:, k], label=f"X_{k}")
            plt.plot(xdot[:, k], label=f"xdot_{k}")
        plt.plot(y[:, 0], label="y")
        plt.plot(u, label="u")
        plt.legend()
        plt.show()
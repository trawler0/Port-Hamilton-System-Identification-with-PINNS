import numpy as np
from scipy.integrate import odeint, solve_ivp
import torch
from utils import sample_initial_states, multi_sin_signal, get_uniform_white_noise, get_noise_bound
from functools import partial
from tqdm import tqdm
import mlflow

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
        t_ = np.linspace(0, self.simulation_time, self.num_steps)
        dynamics = partial(self.system_dynamics, u=u)
        if u is None:
            u_out = np.zeros((self.num_steps, 1))
        else:
            u_out = np.stack([u(t) for t in t_])
        return odeint(dynamics, x0, t_), u_out, u

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
            u = self.generate_signal(seed=j) if self.generate_signal is not None else None
            X, u, signal = self.generate_trajectory(x0, u)
            #xdot = (X[1:] - X[:-1]) / dt      -    this is not any good
            xdot = self.__call__(X[:-1], u[:-1])
            y = np.stack([self.y(x) for x in X])
            trajectories.append((X, u, y, signal))
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
        return 2 * self.M @ x

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

class MultiMassSpringDamper(BaseDataGenerator):

    def __init__(self, N, G_, masses, spring_constants, damping, simulation_time, num_steps, generate_signal=None):
        """
        N: number of mass-spring-damper units.
        masses: array-like of length N containing masses.
        spring_constants: array-like of length N containing spring constants.
        damping: scalar damping coefficient or array-like of length N
        """
        super().__init__(simulation_time, num_steps, generate_signal)
        self.N = N
        self.G_ = G_
        self.masses = np.asarray(masses)
        self.spring_constants = np.asarray(spring_constants)
        self.damping = np.asarray(damping) if hasattr(damping, '__len__') else damping * np.ones(N)

        M_diag = []
        for i in range(N):
            M_diag.extend([spring_constants[i] / 2, 1 / (2 * masses[i])])
        self.M = np.diag(M_diag)


    def H(self, x):
        return x @ self.M @ x / 2

    def grad_H(self, x):
        return self.M @ x

    def J(self, x):
        J_matrix = np.zeros((2*self.masses.size, 2*self.masses.size))
        for i in range(self.masses.size):
            idx = 2*i
            J_matrix[idx, idx+1] = 1
            J_matrix[idx+1, idx] = -1
            if i != self.masses.size - 1:
                J_matrix[idx+1, idx+2] = 1
                J_matrix[idx+2, idx+1] = -1
        return J_matrix

    def R(self, x):
        R_matrix = np.zeros((2*self.masses.size, 2*self.masses.size))
        for i in range(self.masses.size):
            idx = 2*i + 1
            R_matrix[idx, idx] = self.damping[i] / self.masses[i]
        return R_matrix

    def G(self, x):
        return self.G_

    def u(self, t, *args):
        return np.zeros(self.G_.shape[1])


def simple_experiment(name, simulation_time, num_steps, amplitude, f0, **kwargs):
    if name == "spring":
        G = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
        masses = kwargs.pop("masses", (1., 1.5))
        spring_constants = kwargs.pop("spring_constants", (1., .1))
        damping = kwargs.pop("damping", 2.)
        if mlflow.active_run():
            mlflow.log_params({"data_masses": masses, "data_spring_constants": spring_constants, "data_damping": damping, "data_G": G})
        get_u = partial(multi_sin_signal, n_signals=2, amplitude=amplitude, f_0=f0)
        return CoupledSpringMassDamper(G, masses, spring_constants, damping, simulation_time, num_steps, get_u)
    elif name == "ball":
        m = kwargs.pop("m", .1)
        R = kwargs.pop("R", .1)
        c = kwargs.pop("c", 1.)
        G = kwargs.pop("G", np.array([[0], [0], [1]]))
        if mlflow.active_run():
            mlflow.log_params({"data_m": m, "data_R": R, "data_c": c, "data_G": G})
        get_u = partial(multi_sin_signal, amplitude=amplitude, f_0=f0)
        return MagneticBall(m, R, c, G, simulation_time, num_steps, get_u)
    elif name == "motor":
        J_m = kwargs.pop("J_m", 0.012)
        L = kwargs.pop("L", 0.0038)
        beta = kwargs.pop("beta", 0.0026)
        r = kwargs.pop("r", 0.225)
        Phi = kwargs.pop("Phi", 0.17)
        G = kwargs.pop("G", np.array([[1, 0], [0, 1], [0, 0]]))
        if mlflow.active_run():
            mlflow.log_params({"data_J_m": J_m, "data_L": L, "data_beta": beta, "data_r": r, "data_Phi": Phi})
        get_u = partial(multi_sin_signal, n_signals=2, amplitude=amplitude, f_0=f0)
        return PMSM(J_m, L, beta, r, Phi, G, simulation_time, num_steps, get_u)
    elif name == "spring_multi_mass":
        N = kwargs.pop("N", 8)
        masses = kwargs.pop("masses", [1.0 for _ in range(N)])
        spring_constants = kwargs.pop("spring_constants", [1.0 for _ in range(N)])
        damping = kwargs.pop("damping", 0.5)

        G = np.zeros((2 * N, N))
        for i in range(N):
            G[2 * i, i] = 1

        if mlflow.active_run():
            mlflow.log_params(
                {"data_masses": masses, "data_spring_constants": spring_constants, "data_damping": damping,
                 "data_G": G})

        get_u = partial(multi_sin_signal, n_signals=N, amplitude=amplitude, f_0=f0)
        return MultiMassSpringDamper(N, G, masses, spring_constants, damping, simulation_time, num_steps, get_u)
    else:
        raise NotImplementedError("Experiment not implemented")

def dim_bias_scale_sigs(name):
    if name == "spring":
        DIM = 4
        scale = np.array([1., 1., 1., 1.])
        bias = np.array([-0.5, -0.5, -0.5, -0.5])
        sigs = 2
        amplitude_train = 0.4
        f0_train = .1
        amplitude_val = 0.4
        f0_val = .1
    elif name == "ball":
        DIM = 3
        scale = np.array([[2.5, .4, 7.]])
        bias = np.array([[-.5, -.2, -3.]])
        sigs = 1
        amplitude_train = 1.
        f0_train = .1
        amplitude_val = 1.
        f0_val = .1
    elif name == "motor":
        DIM = 3
        scale = np.array([[1, 1, 2]])
        bias = np.array([[-.5, -.5, -1]])
        sigs = 2
        amplitude_train = 5
        f0_train = .1
        amplitude_val = 5
        f0_val = .1
    elif name == "spring_multi_mass":
        DIM = 2 * 8
        scale = np.ones(DIM)
        bias = np.zeros(DIM)
        sigs = 8
        amplitude_train = 0.4
        f0_train = 0.1
        amplitude_val = 0.4
        f0_val = 0.1
    else:
        raise ValueError("Unknown problem")
    return DIM, scale, bias, sigs, amplitude_train, f0_train, amplitude_val, f0_val

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dim, scale, bias, sigs, amplitude_train, f0_train, amplitude_val, f0_val = dim_bias_scale_sigs("spring_multi_mass")
    generator = simple_experiment("spring_multi_mass", 20, 1000, amplitude_train, f0_train)
    X0 = sample_initial_states(100, 16, {"identifies": "uniform", "seed": 41, "scale": scale, "bias": bias})
    X, u, xdot, y, trajectories = generator.get_data(X0)
    a = get_noise_bound(u, 5)

    power_X = np.mean(X ** 2)
    power_u = np.mean(u ** 2)

    xdot_hat = generator(X, u)
    print(xdot.shape, xdot_hat.shape)
    std = np.std(xdot, axis=0, keepdims=True)
    print(std)
    print(np.mean(np.abs(xdot_hat - xdot) / std), np.mean(np.abs(xdot_hat - xdot)))

    for j in range(4, 8):
        X, u, y, _ = trajectories[j]
        print(X[0])
        dt = 1 / 100
        xdot = (X[1:] - X[:-1]) / dt
        X = X[:-1]
        u = u[:-1]

        fig, ax = plt.subplots(X.shape[-1] + u.shape[-1], 1, figsize=(10, 10))
        for i in range(X.shape[-1]):
            ax[i].plot(X[:, i], label=f"X_{i}", color="red")
            #ax[i].plot(xdot[:, i], label=f"Xdot_{i}", color="blue")
        for i in range(u.shape[-1]):
            ax[i + X.shape[-1]].plot(u[:, i], label=f"u_{i}", color="green")
        plt.show()
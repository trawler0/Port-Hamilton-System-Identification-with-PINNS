import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

# --- Load data ---
data = np.load("results.npy", allow_pickle=True)
mn = [
    ("spring_multi_mass", "spring_chain"),
    ("spring_multi_mass", "default"),
    ("spring_multi_mass", "baseline"),
    ("ball", "affine"),
    ("ball", "default"),
    ("motor", "baseline"),
    ("motor", "affine"),
    ("spring", "default"),
    ("spring", "affine"),
    ("spring", "baseline")
]
data = data.item()
method, name = mn[6]
print("Selected:", method, name)

mae = data[f"mae_{name}_{method}"]
accurate_time = data[f"acc_{name}_{method}"][:, 2]
compute = data[f"compute_{name}_{method}"]
trajectories = data[f"traj_{name}_{method}"]
sizes = data[f"sizes_{name}_{method}"]

# --- Filter function for Pareto envelope ---
def filter(compute, metric, descending=True):
    idx = np.argsort(compute)
    c = compute[idx]
    m = metric[idx]
    out = [(c[0], m[0])]
    for i in range(1, len(c)):
        criterion = m[i] < out[-1][1] if descending else m[i] > out[-1][1]
        if criterion:
            out.append([c[i], m[i]])
    c = np.array(out)[:, 0]
    m = np.array(out)[:, 1]
    return c, m

# --- MAE Scaling ---
def mae_scaling(C_raw, M_raw):
    C, M = filter(C_raw, M_raw)
    C = np.log10(C)
    M = np.log10(M)

    def scaling_law_log(C, a, b, c, d):
        return np.log10(a + b * (10**C + c)**d)

    C_norm = (C - np.min(C)) / (np.max(C) - np.min(C))

    p0 = [np.mean(10**M), 1.0, 1.0, -1.0]
    bounds = ([1e-10, 1e-10, 0, -6], [np.inf, np.inf, np.inf, -0.3])  # <- was -0.1

    params, _ = curve_fit(
        scaling_law_log, C, M,
        p0=p0,
        bounds=bounds,
        maxfev=50000
    )
    a, b, c, d = params
    print("MAE fit parameters:", params)

    C_fit = np.linspace(min(C), max(C), 500)
    M_fit = scaling_law_log(C_fit, a, b, c, d)
    return 10**C, 10**M, 10**C_fit, 10**M_fit

def acc_scaling(C_raw, A):
    # Remove very low values
    C_raw, A = C_raw[A > 1e-3], A[A > 1e-3]
    C, A = filter(C_raw, A, descending=False)
    C = np.log10(C)

    def acc_law(C, a, b, c, d, e):
        logistic = a / (1 + b * (10**C + c)**(-d))
        return np.maximum(0, logistic - e)

    # Safe initial guesses
    a0 = np.max(A)
    b0 = 1.0
    c0 = 1.0
    d0 = 1.0
    e0 = np.min(A) * 0.9
    p0 = [a0, b0, c0, d0, e0]

    # Bounds for stability
    bounds = (
        [0, 1e-6, 0, 0.1, 0],       # lower bounds: all positive, d > 0 to ensure growth
        [10*a0, 1e3, 1e6, 10, a0]   # upper bounds
    )

    C_norm = (C - np.min(C)) / (np.max(C) - np.min(C))
    sigma = 0.1 + 10 * (1 - C_norm)**2  # emphasize high-compute region

    params, _ = curve_fit(
        acc_law, C, A,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        loss='soft_l1',
        f_scale=0.1,
        maxfev=100000
    )
    a, b, c, d, e = params
    print("Accurate time fit parameters:", params)

    C_fit = np.linspace(min(C), max(C), 500)
    A_fit = acc_law(C_fit, a, b, c, d, e)
    return 10**C, A, 10**C_fit, A_fit


# --- Plot ---
fig, ax = plt.subplots(1, 2, figsize=(15, 15))

# MAE Plot
C_mae, M_mae, C_fit, M_fit = mae_scaling(compute, mae)
ax[0].scatter(C_mae, M_mae, label="Envelope", color="blue")
ax[0].plot(C_fit, M_fit, label="MAE Fit", color="black")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_title("MAE vs Compute")
ax[0].set_xlabel("Compute")
ax[0].set_ylabel("MAE")
ax[0].legend()

# Accurate Time Plot
ax[1].scatter(compute, accurate_time, c=np.log(trajectories), cmap='coolwarm', edgecolor='k', s=sizes**2/16)
C_acc, A_acc, C_fit_acc, A_fit = acc_scaling(compute, accurate_time)
ax[1].plot(C_fit_acc, A_fit, label="Accurate Time Fit", linestyle="--", color="green")
ax[1].set_xscale("log")
ax[1].set_title("Accurate Time vs Compute")
ax[1].set_xlabel("Compute")
ax[1].set_ylabel("Accurate Time")
ax[1].legend()

plt.tight_layout()
plt.show()

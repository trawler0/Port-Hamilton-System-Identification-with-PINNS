import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import string  # for subplot labels (A, B, C, D, ...)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
plt.rcParams.update({'font.size': 18})  # or any size you want

MM = {
    "spring_chain": "Sparse",
    "default": "Port-Hamilton",
    "baseline": "General",
    "affine": "Input-Affine",
}
DD = {
    "spring_multi_mass": "Spring-Chain",
    "spring": "Spring",
    "motor": "Motor",
    "ball": "Ball",
}

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
idxs = [
    0, 1, 2, 3, 2, 4, 1, 2, 0, 4
]
C_fits = []
M_fits = []
C_fits_mse = []
MS_fits = []
C_fits_acc = []
A_fits = []
params_maes = []
params_mses = []
params_accs = []
data = data.item()
for i, (method, name) in enumerate(mn):

    mae = data[f"mae_{name}_{method}"]
    mse = data[f"mse_{name}_{method}"]
    accurate_time = data[f"acc_{name}_{method}"][:, idxs[i]]
    compute = data[f"compute_{name}_{method}"]
    trajectories = data[f"traj_{name}_{method}"]
    sizes = data[f"sizes_{name}_{method}"]

    def fit_compute(compute, mae, mse, accurate_time):
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

        def mae_scaling(C_raw, M_raw):
            C, M = filter(C_raw, M_raw)
            C = np.log10(C)
            M = np.log10(M)

            def scaling_law_log(C, a, b, c, d):
                return np.log10(b * (10**C + c)**d)

            p0 = [np.mean(10**M), 1.0, 1.0, -1.0]
            bounds = ([1e-10, 1e-10, 0, -6], [np.inf, np.inf, np.inf, -0.3])  # <- was -0.1

            log_C = np.log(C)
            log_C = (log_C - np.min(log_C)) / (np.max(log_C) - np.min(log_C))
            sigma = 0.1 + log_C
            params, _ = curve_fit(
                scaling_law_log, C, M,
                p0=p0,
                sigma=sigma,
                bounds=bounds,
                maxfev=50000
            )
            a, b, c, d = params

            C_fit = np.linspace(min(C), max(C), 500)
            M_fit = scaling_law_log(C_fit, a, b, c, d)
            return 10**C, 10**M, 10**C_fit, 10**M_fit, params

        def acc_scaling(C_raw, A):
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

            log_C = np.log(C)
            log_C = (log_C - np.min(log_C)) / (np.max(log_C) - np.min(log_C))
            sigma = 0.1 + 1 * log_C

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

            C_fit = np.linspace(min(C), max(C), 500)
            A_fit = acc_law(C_fit, a, b, c, d, e)
            return 10**C, A, 10**C_fit, A_fit, params

        C_mae, M_mae, C_fit, M_fit, params_mae = mae_scaling(compute, mae)
        C_mse, M_mse, C_fit_mse, M_fit_mse, params_mse = mae_scaling(compute, mse)
        C_acc, A_acc, C_fit_acc, A_fit, params_acc = acc_scaling(compute, accurate_time)
        return C_fit, M_fit, C_fit_mse, M_fit_mse, C_fit_acc, A_fit, params_mae, params_mse, params_acc

    C_fit, M_fit, C_fit_mse, M_fit_mse, C_fit_acc, A_fit, params_mae, params_mse, params_acc = fit_compute(compute, mae, mse, accurate_time)

    C_fits.append(C_fit)
    M_fits.append(M_fit)

    C_fits_mse.append(C_fit_mse)
    MS_fits.append(M_fit_mse)

    C_fits_acc.append(C_fit_acc)
    A_fits.append(A_fit)

    params_maes.append(np.array(params_mae))
    params_accs.append(np.array(params_acc))
    params_mses.append(np.array(params_mse))

C_fits = np.stack(C_fits)
M_fits = np.stack(M_fits)

C_fits_mse = np.stack(C_fits_mse)
MS_fits = np.stack(MS_fits)

C_fits_acc = np.stack(C_fits_acc)
A_fits = np.stack(A_fits)

params_maes = np.stack(params_maes)
params_accs = np.stack(params_accs)
params_mses = np.stack(params_mses)


def low_data_prior(data, mn, N=10):
    compute0 = data[f"compute_{mn[0][1]}_{mn[0][0]}"]
    compute1 = data[f"compute_{mn[1][1]}_{mn[1][0]}"]
    compute2 = data[f"compute_{mn[2][1]}_{mn[2][0]}"]

    mae0 = data[f"mae_{mn[0][1]}_{mn[0][0]}"]
    mae1 = data[f"mae_{mn[1][1]}_{mn[1][0]}"]
    mae2 = data[f"mae_{mn[2][1]}_{mn[2][0]}"]

    acc0 = data[f"acc_{mn[0][1]}_{mn[0][0]}"][:, idxs[0]]
    acc1 = data[f"acc_{mn[1][1]}_{mn[1][0]}"][:, idxs[1]]
    acc2 = data[f"acc_{mn[2][1]}_{mn[2][0]}"][:, idxs[2]]

    trajectories0 = data[f"traj_{mn[0][1]}_{mn[0][0]}"]
    trajectories1 = data[f"traj_{mn[1][1]}_{mn[1][0]}"]
    trajectories2 = data[f"traj_{mn[2][1]}_{mn[2][0]}"]

    # Filter to up to 10 trajectories
    idx0 = (trajectories0 <= N)
    idx1 = (trajectories1 <= N)
    idx2 = (trajectories2 <= N)

    compute0 = compute0[idx0]
    compute1 = compute1[idx1]
    compute2 = compute2[idx2]

    mae0 = mae0[idx0]
    mae1 = mae1[idx1]
    mae2 = mae2[idx2]

    acc0 = acc0[idx0]
    acc1 = acc1[idx1]
    acc2 = acc2[idx2]

    trajectories0 = trajectories0[idx0]
    trajectories1 = trajectories1[idx1]
    trajectories2 = trajectories2[idx2]

    trajectories0 = trajectories0 / np.max(trajectories0) * 10
    trajectories1 = trajectories1 / np.max(trajectories1) * 10
    trajectories2 = trajectories2 / np.max(trajectories2) * 10

    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 15))

    # Scatter plots (with some alpha for transparency)
    ax.scatter(
        compute0, mae0,
        label=MM[mn[0][1]],
        color="blue",
        s=trajectories0 * 10,
        alpha=0.7
    )
    ax.scatter(
        compute1, mae1,
        label=MM[mn[1][1]],
        color="red",
        s=trajectories1 * 10,
        alpha=0.7
    )
    ax.scatter(
        compute2, mae2,
        label=MM[mn[2][1]],
        color="green",
        s=trajectories2 * 10,
        alpha=0.7
    )

    # Log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Axis labels
    ax.set_xlabel("Compute")
    ax.set_ylabel("Mean Absolute Error (MAE)")

    # Show grid
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Legend
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"figures/low_data_prior_{N}.jpg", bbox_inches='tight')
    # plt.show()


def mae_examples_nicer(data, mn, C, M, params, metric="mae"):
    all_trajectories = np.concatenate([
        data[f"traj_{mn[i][1]}_{mn[i][0]}"] for i in range(4)
    ])
    vmin, vmax = all_trajectories.min(), all_trajectories.max()

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    fig.subplots_adjust(right=0.85, wspace=0.25, hspace=0.3)

    labels = list(string.ascii_uppercase)

    scatter_kwargs = dict(
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.7,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    def sf(x):
        return f"{x:.2g}"

    def add_annotation(ax, title_str, formula_str):
        text_str = rf"{title_str}"
        ax.text(0.03, 0.1, text_str, transform=ax.transAxes,
                ha='left', va='bottom')
        #ax.text(0.03, 0.05, f"${formula_str}$", transform=ax.transAxes,
        #        ha='left', va='bottom', fontsize=9)

    for i in range(4):
        compute = data[f"compute_{mn[i][1]}_{mn[i][0]}"]
        mae = data[f"{metric}_{mn[i][1]}_{mn[i][0]}"]
        trajectories = data[f"traj_{mn[i][1]}_{mn[i][0]}"]
        sizes = data[f"sizes_{mn[i][1]}_{mn[i][0]}"]

        criterion = compute < 1e5
        idx = np.where(criterion)[0]
        sc = axes[i].scatter(
            compute[idx], mae[idx],
            c=trajectories[idx],
            s=(sizes[idx] ** 2) / 16,
            **scatter_kwargs
        )
        criterion = C[i] < 1e5
        idx = np.where(criterion)[0]
        axes[i].plot(C[i][idx], M[i][idx])
        a, b, c_, d = params[i]
        formula = rf"f(C) = {sf(a)} + {sf(b)}(C + {sf(c_)})^{{{sf(d)}}}"
        add_annotation(axes[i], f"{MM[mn[i][1]]} - {DD[mn[i][0]]}", formula)

        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].grid(True, which='both', linestyle='--', alpha=0.2)
        axes[i].text(0.01, 1.02, f"({labels[i]})", transform=axes[i].transAxes,
                     ha='left', va='bottom', fontweight='bold')

    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])
    sm = mpl.cm.ScalarMappable(cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Trajectory Count", labelpad=10)
    ticks = np.geomspace(vmin, vmax, num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$10^{{{int(np.log10(t))}}}$" for t in ticks])

    plt.savefig(f"figures/{metric}_examples.jpg", bbox_inches='tight')

def acc_examples_nicer(data, mn, C, M, params, idxs):
    all_trajectories = np.concatenate([
        data[f"traj_{mn[i][1]}_{mn[i][0]}"] for i in range(4)
    ])
    vmin, vmax = all_trajectories.min(), all_trajectories.max()

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    fig.subplots_adjust(right=0.85, wspace=0.25, hspace=0.3)

    labels = list(string.ascii_uppercase)

    scatter_kwargs = dict(
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.7,
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    def sf(x):
        """Return LaTeX string with scientific notation if needed."""
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        base = x / 10 ** exp
        if exp == 0:
            return f"{base:.2g}"
        return rf"{base:.2g} \times 10^{{{exp}}}"

    def add_annotation(ax, title_str, formula_str):
        text_str = rf"{title_str}"
        ax.text(0.03, 0.95, text_str, transform=ax.transAxes,
                ha='left', va='bottom')
        #ax.text(0.03, 0.9, formula_str, transform=ax.transAxes,
        #        ha='left', va='bottom', fontsize=9)

    for i in range(4):
        compute = data[f"compute_{mn[i][1]}_{mn[i][0]}"]
        acc = data[f"acc_{mn[i][1]}_{mn[i][0]}"][:, idxs[i]]
        trajectories = data[f"traj_{mn[i][1]}_{mn[i][0]}"]
        sizes = data[f"sizes_{mn[i][1]}_{mn[i][0]}"]

        criterion = compute < 1e5
        idx = np.where(criterion)[0]
        sc = axes[i].scatter(
            compute[idx], acc[idx],
            c=trajectories[idx],
            s=(sizes[idx] ** 2) / 16,
            **scatter_kwargs
        )
        criterion = C[i] < 1e5
        idx = np.where(criterion)[0]
        axes[i].plot(C[i][idx], M[i][idx])
        a, b, c_, d, e = params[i]
        a_str = sf(a)
        b_str = sf(b)
        c_str = sf(c_)
        d_str = sf(d)
        e_str = sf(e)

        # Build the LaTeX formula string
        formula = (
            r"$f(C) = \mathrm{ReLU}\left("
            rf"\frac{{{a_str}}}{{1 + {b_str} \cdot (x + {c_str})^{{{d_str}}}}}"
            rf" - {e_str}"
            r"\right)$"
        )
        add_annotation(axes[i], f"{MM[mn[i][1]]} - {DD[mn[i][0]]}", formula)

        axes[i].set_xscale('log')
        axes[i].grid(True, which='both', linestyle='--', alpha=0.2)
        axes[i].text(0.01, 1.02, f"({labels[i]})", transform=axes[i].transAxes,
                     ha='left', va='bottom', fontweight='bold')

    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])
    sm = mpl.cm.ScalarMappable(cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Trajectory Count", labelpad=10)
    ticks = np.geomspace(vmin, vmax, num=5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"$10^{{{int(np.log10(t))}}}$" for t in ticks])

    plt.savefig(f"figures/acc_examples.jpg", bbox_inches='tight')

#low_data_prior(data, mn[-3:])
#low_data_prior(data, mn[:3], N=100)
mae_examples_nicer(data, [mn[2], mn[6], mn[4], mn[7]], [C_fits[2], C_fits[6], C_fits[4], C_fits[7]],
            [M_fits[2], M_fits[6], M_fits[4], M_fits[7]], [params_maes[2], params_maes[6], params_maes[4], params_maes[7]])
mae_examples_nicer(data, [mn[2], mn[5], mn[9], mn[2]], [C_fits_mse[2], C_fits_mse[5], C_fits_mse[9], C_fits_mse[2]],
            [MS_fits[2], MS_fits[5], MS_fits[9], MS_fits[2]], [params_maes[2], params_maes[5], params_maes[9], params_maes[2]], metric="mse")

acc_examples_nicer(data, [mn[5], mn[8], mn[3], mn[6]], [C_fits_acc[5], C_fits_acc[8], C_fits_acc[3], C_fits_acc[6]],
            [A_fits[5], A_fits[8], A_fits[3], A_fits[6]], [params_accs[5], params_accs[8], params_accs[3], params_accs[6]],
                [idxs[5], idxs[8], idxs[3], idxs[6]])


fig, ax = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Choose which two method-name pairs to compare
n = [3, 4]
compute_all = np.concatenate([
    data[f"compute_{name}_{method}"] for method, name in [mn[n[0]], mn[n[1]]]
])
vmin, vmax = compute_all.min(), compute_all.max()

# Loop over the two selected models
for j, (method, name) in enumerate([mn[n[0]], mn[n[1]]]):
    trajectories = data[f"traj_{name}_{method}"]
    nmae = data[f"mae_{name}_{method}"]
    nmse = data[f"mse_{name}_{method}"]
    acc = data[f"acc_{name}_{method}"][:, idxs[n[j]]]
    compute = data[f"compute_{name}_{method}"]
    sizes = data[f"sizes_{name}_{method}"]

    # Fit curves (assumes fit_compute returns proper curves and params)
    log_traj = np.log10(trajectories)
    log_traj = (log_traj - np.min(log_traj)) / (np.max(log_traj) - np.min(log_traj))
    out = fit_compute(trajectories, nmae, nmse, acc)
    T_fit, M_fit, T_fit_mse, M_fit_mse, T_fit_acc, A_fit, params_mae, params_mse, params_acc = out

    # --- MAE subplot ---
    sc1 = ax[j, 0].scatter(
        trajectories, nmae,
        c=compute, cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=0.8, s=sizes**2 / 30
    )
    ax[j, 0].plot(T_fit, M_fit, color="black")
    ax[j, 0].set_title(f"{DD[method]} — {MM[name]}", fontsize=11)
    ax[j, 0].set_ylabel("MAE")
    ax[j, 0].set_xlabel("Trajectory Count")
    ax[j, 0].set_xscale("log")
    ax[j, 0].set_yscale("log")
    ax[j, 0].grid(True, which='both', linestyle='--', alpha=0.3)

    # --- Accurate Time subplot ---
    sc2 = ax[j, 1].scatter(
        trajectories, acc,
        c=compute, cmap='coolwarm', norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=0.8, s=sizes**2 / 30
    )
    ax[j, 1].plot(T_fit_acc, A_fit, color="black")
    ax[j, 1].set_title(f"{DD[method]} — {MM[name]}", fontsize=11)
    ax[j, 1].set_ylabel("Accurate Time")
    ax[j, 1].set_xlabel("Trajectory Count")
    ax[j, 1].set_xscale("log")
    ax[j, 1].grid(True, which='both', linestyle='--', alpha=0.3)

# --- Shared colorbar for compute time ---
cbar = fig.colorbar(
    ScalarMappable(norm=LogNorm(vmin=vmin, vmax=vmax), cmap='coolwarm'),
    ax=ax.ravel().tolist(),
    shrink=0.95,
    label='Compute Time'
)

# --- Centered headline ---
fig.suptitle("Scaling of Error and Accuracy with Data and Compute", fontsize=16, y=1.03)

# Save to file
plt.savefig("figures/data_scaling.jpg", bbox_inches='tight')
# plt.show()
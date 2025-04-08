import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import string  # for subplot labels (A, B, C, D, ...)

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
    0, 1, 2, 3, 4, 3, 1, 2, 4, 4
]
C_fits = []
M_fits = []
C_fits_acc = []
A_fits = []
params_maes = []
params_accs = []
data = data.item()
for i, (method, name) in enumerate(mn):

    mae = data[f"mae_{name}_{method}"]
    accurate_time = data[f"acc_{name}_{method}"][:, idxs[i]]
    print(np.max(accurate_time))
    compute = data[f"compute_{name}_{method}"]
    trajectories = data[f"traj_{name}_{method}"]
    sizes = data[f"sizes_{name}_{method}"]

    def fit_compute(compute, mae, accurate_time, hard_a=False):
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
                return np.log10(0 if hard_a else a + b * (10**C + c)**d)

            p0 = [np.mean(10**M), 1.0, 1.0, -1.0]
            bounds = ([1e-10, 1e-10, 0, -6], [np.inf, np.inf, np.inf, -0.3])  # <- was -0.1

            params, _ = curve_fit(
                scaling_law_log, C, M,
                p0=p0,
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

            C_fit = np.linspace(min(C), max(C), 500)
            A_fit = acc_law(C_fit, a, b, c, d, e)
            return 10**C, A, 10**C_fit, A_fit, params

        C_mae, M_mae, C_fit, M_fit, params_mae = mae_scaling(compute, mae)
        C_acc, A_acc, C_fit_acc, A_fit, params_acc = acc_scaling(compute, accurate_time)
        return C_fit, M_fit, C_fit_acc, A_fit, params_mae, params_acc


    C_fit, M_fit, C_fit_acc, A_fit, params_mae, params_acc = fit_compute(compute, mae, accurate_time)
    C_fits.append(C_fit)
    M_fits.append(M_fit)
    C_fits_acc.append(C_fit_acc)
    A_fits.append(A_fit)
    params_maes.append(np.array(params_mae))
    params_accs.append(np.array(params_acc))
C_fits = np.stack(C_fits)
M_fits = np.stack(M_fits)
C_fits_acc = np.stack(C_fits_acc)
A_fits = np.stack(A_fits)
params_maes = np.stack(params_maes)
params_accs = np.stack(params_accs)


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
        label=mn[0][1],
        color="blue",
        s=trajectories0 * 10,
        alpha=0.7
    )
    ax.scatter(
        compute1, mae1,
        label=mn[1][1],
        color="red",
        s=trajectories1 * 10,
        alpha=0.7
    )
    ax.scatter(
        compute2, mae2,
        label=mn[2][1],
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


def mae_examples_nicer(data, mn, C, M, params):
    """
    Plot MAE vs. Compute for four different models (mn[0]...mn[3]).

    - Color indicates log(# Trajectories)
    - Marker size indicates 'sizesX'
    - Each subplot shows a scaling law formula.
    - We add method+model info from mn[i].
    """

    # --- Data extraction ---
    compute0 = data[f"compute_{mn[0][1]}_{mn[0][0]}"]
    compute1 = data[f"compute_{mn[1][1]}_{mn[1][0]}"]
    compute2 = data[f"compute_{mn[2][1]}_{mn[2][0]}"]
    compute3 = data[f"compute_{mn[3][1]}_{mn[3][0]}"]

    mae0 = data[f"mae_{mn[0][1]}_{mn[0][0]}"]
    mae1 = data[f"mae_{mn[1][1]}_{mn[1][0]}"]
    mae2 = data[f"mae_{mn[2][1]}_{mn[2][0]}"]
    mae3 = data[f"mae_{mn[3][1]}_{mn[3][0]}"]

    trajectories0 = data[f"traj_{mn[0][1]}_{mn[0][0]}"]
    trajectories1 = data[f"traj_{mn[1][1]}_{mn[1][0]}"]
    trajectories2 = data[f"traj_{mn[2][1]}_{mn[2][0]}"]
    trajectories3 = data[f"traj_{mn[3][1]}_{mn[3][0]}"]

    sizes0 = data[f"sizes_{mn[0][1]}_{mn[0][0]}"]
    sizes1 = data[f"sizes_{mn[1][1]}_{mn[1][0]}"]
    sizes2 = data[f"sizes_{mn[2][1]}_{mn[2][0]}"]
    sizes3 = data[f"sizes_{mn[3][1]}_{mn[3][0]}"]

    # Combine all trajectories for a consistent log-color range
    all_trajectories = np.concatenate([trajectories0, trajectories1, trajectories2, trajectories3])
    all_logs = np.log(all_trajectories)
    vmin, vmax = all_logs.min(), all_logs.max()

    # Figure & Subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    fig.subplots_adjust(right=0.85, wspace=0.25, hspace=0.3)

    labels = list(string.ascii_uppercase)  # For subplot labels: (A), (B), (C), (D)

    scatter_kwargs = dict(
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.7,
        vmin=vmin,
        vmax=vmax
    )

    def add_annotation(ax, title_str, formula_str):
        """
        Place the method/model info and formula at the bottom-left corner of the axes.
        title_str = e.g. "method_model"
        formula_str = e.g. "0.45 + 0.34*(x+0.12)^-0.22"
        """
        text_str = f"{title_str}\n{formula_str}"
        ax.text(
            0.03, 0.05,  # near bottom-left corner
            text_str,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=9
        )

    # Helper to format big numbers in scientific notation (e.g. 1.0e+03)
    # using 2 significant figures. Adjust if you need more or fewer.
    def sf(x):
        return f"{x:.2g}"

    # --- Subplot 1 ---
    sc0 = axes[0].scatter(
        compute0, mae0,
        c=np.log(trajectories0),
        s=(sizes0 ** 2) / 16,
        **scatter_kwargs
    )
    axes[0].plot(C[0], M[0])
    a0, b0, c0_, d0 = params[0]
    method_name_0 = f"{mn[0][1]}_{mn[0][0]}"
    # Build formula with scientific notation
    formula0 = f"{sf(a0)} + {sf(b0)}*(x+{sf(c0_)})^{sf(d0)}"
    add_annotation(axes[0], method_name_0, formula0)

    # --- Subplot 2 ---
    sc1 = axes[1].scatter(
        compute1, mae1,
        c=np.log(trajectories1),
        s=(sizes1 ** 2) / 16,
        **scatter_kwargs
    )
    axes[1].plot(C[1], M[1])
    a1, b1, c1_, d1 = params[1]
    method_name_1 = f"{mn[1][1]}_{mn[1][0]}"
    formula1 = f"{sf(a1)} + {sf(b1)}*(x+{sf(c1_)})^{sf(d1)}"
    add_annotation(axes[1], method_name_1, formula1)

    # --- Subplot 3 ---
    sc2 = axes[2].scatter(
        compute2, mae2,
        c=np.log(trajectories2),
        s=(sizes2 ** 2) / 16,
        **scatter_kwargs
    )
    axes[2].plot(C[2], M[2])
    a2, b2, c2_, d2 = params[2]
    method_name_2 = f"{mn[2][1]}_{mn[2][0]}"
    formula2 = f"{sf(a2)} + {sf(b2)}*(x+{sf(c2_)})^{sf(d2)}"
    add_annotation(axes[2], method_name_2, formula2)

    # --- Subplot 4 ---
    sc3 = axes[3].scatter(
        compute3, mae3,
        c=np.log(trajectories3),
        s=(sizes3 ** 2) / 16,
        **scatter_kwargs
    )
    axes[3].plot(C[3], M[3])
    a3, b3, c3_, d3 = params[3]
    method_name_3 = f"{mn[3][1]}_{mn[3][0]}"
    formula3 = f"{sf(a3)} + {sf(b3)}*(x+{sf(c3_)})^{sf(d3)}"
    add_annotation(axes[3], method_name_3, formula3)

    # Format Each Subplot
    for i, ax in enumerate(axes):
        ax.text(
            0.01, 1.02,
            f"({labels[i]})",
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Compute")
        ax.set_ylabel("MAE")
        ax.grid(True, which='both', linestyle='--', alpha=0.2)

    # No labeled line, so no legend needed for the lines.
    # If you still have something to show in the legend, you can do:
    # lines, line_labels = axes[0].get_legend_handles_labels()
    # fig.legend(lines, line_labels, loc='upper center', ncol=4, frameon=False)

    # Color bar on the right (outside)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])
    sm = mpl.cm.ScalarMappable(
        cmap='coolwarm',
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("log(# Trajectories)", labelpad=10)

    plt.savefig(f"figures/mae_examples.jpg", bbox_inches='tight')
    #plt.show()


def acc_examples_nicer(data, mn, C, M, params, idxs):
    """
    Plot ACC vs. Compute for four different models (mn[0]...mn[3]).

    - Color indicates log(# Trajectories)
    - Marker size indicates 'sizesX'
    - Each subplot shows a 'relu(...)' style formula.
    - We add method+model info from mn[i].
    - Large numbers in the formula are displayed in scientific notation.
    """
    # --- Data extraction ---
    compute0 = data[f"compute_{mn[0][1]}_{mn[0][0]}"]
    compute1 = data[f"compute_{mn[1][1]}_{mn[1][0]}"]
    compute2 = data[f"compute_{mn[2][1]}_{mn[2][0]}"]
    compute3 = data[f"compute_{mn[3][1]}_{mn[3][0]}"]

    acc0 = data[f"acc_{mn[0][1]}_{mn[0][0]}"][:, idxs[0]]
    acc1 = data[f"acc_{mn[1][1]}_{mn[1][0]}"][:, idxs[1]]
    acc2 = data[f"acc_{mn[2][1]}_{mn[2][0]}"][:, idxs[2]]
    acc3 = data[f"acc_{mn[3][1]}_{mn[3][0]}"][:, idxs[3]]

    trajectories0 = data[f"traj_{mn[0][1]}_{mn[0][0]}"]
    trajectories1 = data[f"traj_{mn[1][1]}_{mn[1][0]}"]
    trajectories2 = data[f"traj_{mn[2][1]}_{mn[2][0]}"]
    trajectories3 = data[f"traj_{mn[3][1]}_{mn[3][0]}"]

    sizes0 = data[f"sizes_{mn[0][1]}_{mn[0][0]}"]
    sizes1 = data[f"sizes_{mn[1][1]}_{mn[1][0]}"]
    sizes2 = data[f"sizes_{mn[2][1]}_{mn[2][0]}"]
    sizes3 = data[f"sizes_{mn[3][1]}_{mn[3][0]}"]

    # Combine all trajectories for consistent log-color scaling
    all_trajectories = np.concatenate([trajectories0, trajectories1, trajectories2, trajectories3])
    all_logs = np.log(all_trajectories)
    vmin, vmax = all_logs.min(), all_logs.max()

    # Figure & Subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    fig.subplots_adjust(right=0.85, wspace=0.25, hspace=0.3)

    labels = list(string.ascii_uppercase)

    scatter_kwargs = dict(
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.7,
        vmin=vmin,
        vmax=vmax
    )

    # Helper: scientific notation for large numbers
    def sf(x):
        return f"{x:.2g}"

    def add_annotation(ax, title_str, formula_str):
        """
        Place the method/model info and formula in the axes.
        We put it somewhat higher (y=0.8) to avoid interfering with data near the bottom.
        """
        text_str = f"{title_str}\n{formula_str}"
        ax.text(
            0.03, 0.8,
            text_str,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=9
        )

    # --- Subplot 1 ---
    sc0 = axes[0].scatter(
        compute0, acc0,
        c=np.log(trajectories0),
        s=(sizes0 ** 2) / 16,
        **scatter_kwargs
    )
    axes[0].plot(C[0], M[0])
    a0, b0, c0, d0, e0 = params[0]
    method_name_0 = f"{mn[0][1]}_{mn[0][0]}"
    # Use scientific notation in the formula
    formula0 = f"relu({sf(a0)}/(1+{sf(b0)}*(x+{sf(c0)})^{sf(d0)}) - {sf(e0)})"
    add_annotation(axes[0], method_name_0, formula0)

    # --- Subplot 2 ---
    sc1 = axes[1].scatter(
        compute1, acc1,
        c=np.log(trajectories1),
        s=(sizes1 ** 2) / 16,
        **scatter_kwargs
    )
    axes[1].plot(C[1], M[1])
    a1, b1, c1_, d1, e1 = params[1]
    method_name_1 = f"{mn[1][1]}_{mn[1][0]}"
    formula1 = f"relu({sf(a1)}/(1+{sf(b1)}*(x+{sf(c1_)})^{sf(d1)}) - {sf(e1)})"
    add_annotation(axes[1], method_name_1, formula1)

    # --- Subplot 3 ---
    sc2 = axes[2].scatter(
        compute2, acc2,
        c=np.log(trajectories2),
        s=(sizes2 ** 2) / 16,
        **scatter_kwargs
    )
    axes[2].plot(C[2], M[2])
    a2, b2, c2_, d2, e2 = params[2]
    method_name_2 = f"{mn[2][1]}_{mn[2][0]}"
    formula2 = f"relu({sf(a2)}/(1+{sf(b2)}*(x+{sf(c2_)})^{sf(d2)}) - {sf(e2)})"
    add_annotation(axes[2], method_name_2, formula2)

    # --- Subplot 4 ---
    sc3 = axes[3].scatter(
        compute3, acc3,
        c=np.log(trajectories3),
        s=(sizes3 ** 2) / 16,
        **scatter_kwargs
    )
    axes[3].plot(C[3], M[3])
    a3, b3, c3_, d3, e3 = params[3]
    method_name_3 = f"{mn[3][1]}_{mn[3][0]}"
    formula3 = f"relu({sf(a3)}/(1+{sf(b3)}*(x+{sf(c3_)})^{sf(d3)}) - {sf(e3)})"
    add_annotation(axes[3], method_name_3, formula3)

    # Format subplots
    for i, ax in enumerate(axes):
        ax.text(
            0.01, 1.02,
            f"({labels[i]})",
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xscale('log')
        ax.set_xlabel("Compute")
        # If this is truly an accuracy plot, let's label the y-axis "Accuracy"
        # instead of "MAE". Adjust if needed.
        ax.set_ylabel("Accurate Time")
        ax.grid(True, which='both', linestyle='--', alpha=0.2)

    # If you want a legend for the lines (they have no label though), you could add labels
    # and do something like:
    # lines, line_labels = axes[0].get_legend_handles_labels()
    # fig.legend(lines, line_labels, loc='upper center', ncol=4, frameon=False)

    # Color bar for log(# Trajectories)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.6])
    sm = mpl.cm.ScalarMappable(
        cmap='coolwarm',
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("log(# Trajectories)", labelpad=10)

    #plt.show()
    plt.savefig(f"figures/acc_examples.jpg", bbox_inches='tight')


low_data_prior(data, mn[-3:])
low_data_prior(data, mn[:3], N=100)
mae_examples_nicer(data, [mn[2], mn[4], mn[9], mn[7]], [C_fits[2], C_fits[4], C_fits[9], C_fits[7]],
            [M_fits[2], M_fits[4], M_fits[9], M_fits[7]], [params_maes[2], params_maes[4], params_maes[9], params_maes[7]])

acc_examples_nicer(data, [mn[5], mn[8], mn[3], mn[6]], [C_fits_acc[5], C_fits_acc[8], C_fits_acc[3], C_fits_acc[6]],
            [A_fits[5], A_fits[8], A_fits[3], A_fits[6]], [params_accs[5], params_accs[8], params_accs[3], params_accs[6]],
                [idxs[5], idxs[8], idxs[3], idxs[6]])












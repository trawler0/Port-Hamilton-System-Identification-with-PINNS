import mlflow
import numpy as np
from matplotlib import pyplot as plt
import os
from data import simple_experiment, dim_bias_scale_sigs
from utils import sample_initial_states
import torch


#plt.style.use('seaborn-darkgrid')  # You can choose other styles like 'ggplot', 'classic', etc.

if not os.path.exists("results"):
    os.makedirs("results")

colors = {
    "True": "black",
    "baseline": "blue",
    "default": "red",
    "prior": "green",
    "shallow": "orange",
    "shorter_x.1": "purple",
}
line_styles = {
    "True": "-",
    "baseline": "--",
    "default": "--",
    "prior": "--",
    "shallow": "--",
    "shorter_x.1": "--",
}
thickness = {
    "True": 2,
    "baseline": 1,
    "default": 1,
    "prior": 1,
    "shallow": 1,
    "shorter_x.1": 1,
}

def noise_plots():
    experiment = mlflow.get_experiment_by_name("noise")
    runs = mlflow.search_runs(experiment.experiment_id)
    X = []
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params

        dB = params["dB"]

        # Download artifact
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if artifact_file_path.endswith("X_pred.npy"):
                    X_pred = np.load(artifact_file_path)
                    X_true = np.load(artifact_file_path.replace("X_pred", "X"))
                    X.append((dB, X_pred))
    n = X_pred.shape[-1]
    fig, ax = plt.subplots(n, 1, figsize=(30, 30))
    t = np.arange(1000) * 0.01
    for j in range(n):
        for dB, X_pred in X:
            ax[j].plot(t, X_pred[6, :1000, j], label=f"dB={dB}")
        ax[j].plot(t, X_true[6, :1000, j], label="True", color="black")
        ax[j].set_xlabel("Time [s]")
        ax[j].set_ylabel("Trajectory")
        ax[j].grid()
        ax[j].set_title(f"Trajectory {j}")
        ax[j].legend(fontsize=32)
    plt.show()

    plt.savefig(os.path.join("results", "noise.png"))


def plot_scaling():
    experiment = mlflow.get_experiment_by_name("scaling")
    all_runs = mlflow.search_runs(experiment.experiment_id)
    for name in ["ball", "motor", "spring"]:
        runs = all_runs[all_runs["tags.name"] == name]
        baseline = {}
        default = {}
        prior = {}
        for i, run in runs.iterrows():
            run_id = run["run_id"]
            run = mlflow.get_run(run_id)
            params = run.data.params
            metrics = run.data.metrics
            run_name = params["run_name"]
            mae_rel = metrics["mae"]
            mse_rel = metrics["mse"]
            if run_name.startswith(f"{name}_baseline"):
                baseline[int(params["num_trajectories"])] = [mae_rel, mse_rel]
            elif run_name.startswith(f"{name}_default"):
                default[int(params["num_trajectories"])] = [mae_rel, mse_rel]
            elif run_name.startswith(f"{name}_prior"):
                prior[int(params["num_trajectories"])] = [mae_rel, mse_rel]
        assert default.keys() == baseline.keys() == prior.keys()
        sorted_keys = sorted(default.keys())
        mae_rel_baseline = [baseline[k][0] for k in sorted_keys]
        mae_rel_default = [default[k][0] for k in sorted_keys]
        mae_rel_prior = [prior[k][0] for k in sorted_keys]


        # Create the figure and axis with an optimized size
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted from (30, 30) to (12, 8)

        # Plotting the data with markers and increased line width for better visibility
        ax.plot(sorted_keys, mae_rel_baseline, label="Baseline", marker='o', linestyle=":", color=colors["baseline"])
        ax.plot(sorted_keys, mae_rel_default, label="Default", marker='s', linestyle=":", color=colors["default"])
        ax.plot(sorted_keys, mae_rel_prior, label="Prior", marker='^', linestyle=":", color=colors["prior"])

        # Set log scales
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Set title and labels with increased font sizes
        ax.set_xlabel("Number of Trajectories Used for Training", fontsize=16)
        ax.set_ylabel("Normalized MAE", fontsize=16)  # Added y-label for completeness

        # Customize tick parameters for better readability
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)

        # Enable and customize grid
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        # Customize legend with larger font size and appropriate placement
        ax.legend(fontsize=32, loc='best')  # 'best' lets matplotlib decide the optimal location

        # Optional: Tight layout for better spacing
        plt.tight_layout()
        #plt.show()

        plt.savefig(os.path.join("results", f"{name}_scaling.png"))


def recipe():
    experiment = mlflow.get_experiment_by_name("recipe")
    runs = mlflow.search_runs(experiment.experiment_id)
    preds = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if artifact_file_path.endswith("X_pred.npy"):
                    X_pred = np.load(artifact_file_path)
                    X_true = np.load(artifact_file_path.replace("X_pred", "X"))
                    preds[run_name] = X_pred
    n = X_pred.shape[-1]
    idx1 = 3
    state1 = 1
    idx2 = 7
    state2 = 3
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    time = np.arange(len(X_true[idx1, :4000, state1])) * 0.01
    for j, (idx, state) in enumerate([(idx1, state1), (idx2, state2)]):
        ax[j].plot(time, X_true[idx, :4000, state], label="True", color="black", linestyle="-", linewidth=thickness["True"])
        for run_name, X_pred in preds.items():
            ax[j].plot(time, X_pred[idx, :4000, state], label=run_name, color=colors[run_name], linestyle=line_styles[run_name], linewidth=thickness[run_name])
        ax[j].set_xlabel("Time [s]")
        ax[j].set_ylabel("State {}".format(state))
        ax[j].axvline(x=10., color='purple', linestyle='--', linewidth=2)
        ax[j].grid()
        ax[j].legend(fontsize=32)
    #plt.show()
    plt.savefig(os.path.join("results", "recipe.png"))

def compare():
    experiment = mlflow.get_experiment_by_name("compare")
    runs = mlflow.search_runs(experiment.experiment_id)
    preds = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if artifact_file_path.endswith("X_pred.npy"):
                    X_pred = np.load(artifact_file_path)
                    X_true = np.load(artifact_file_path.replace("X_pred", "X"))
                    preds[run_name] = X_pred

    idx1 = 6
    state1 = 1
    idx2 = 19
    state2 = 0
    n = X_pred.shape[-1]
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    for j, (idx, state) in enumerate([(idx1, state1), (idx2, state2)]):
        ax[j].set_xlabel("Time [s]")
        ax[j].set_ylabel("State {}".format(state))
        t = np.arange(4000) * 0.01
        ax[j].plot(t, X_true[idx, :4000, state], label="True", color="black", linestyle="-", linewidth=thickness["True"])
        for run_name, X_pred in preds.items():
            ax[j].plot(t, X_pred[idx, :4000, state], label=run_name[0].upper() + run_name[1:], color=colors[run_name], linestyle=line_styles[run_name], linewidth=thickness[run_name])
        ax[j].axvline(x=10., color='purple', linestyle='--', linewidth=2)
        ax[j].legend(fontsize=32)
        ax[j].grid()
    # plt.show()
    plt.savefig(os.path.join("results", "compare.png"))


def prior_vs_default():
    experiment = mlflow.get_experiment_by_name("prior_vs_default")
    runs = mlflow.search_runs(experiment.experiment_id)
    models = {}
    for i, run in runs.iterrows():
        run_id = run["run_id"]
        run = mlflow.get_run(run_id)
        params = run.data.params
        run_name = params["run_name"]
        name = params["name"]
        artifacts_path = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Example: If artifact is a file, handle it
        for root, _, files in os.walk(artifacts_path):
            for file in files:
                artifact_file_path = os.path.join(root, file)
                if file == "model.pth" or file.endswith(".pt"):
                    # Assuming the model was saved with a specific extension
                    model_uri = f"runs:/{run_id}/model"
                    try:
                        model = mlflow.pytorch.load_model(model_uri)
                        models[run_name] = model
                    except Exception as e:
                        print(f"Failed to load model for run {run_id}: {e}")
        dim, scale, bias, sigs = dim_bias_scale_sigs(name)
        X = sample_initial_states(500, dim, {"identifies": "uniform", "seed": 41})
        generator = simple_experiment(name, 10, 1000)
        grad_H = np.stack([generator.grad_H(x) for x in X], axis=0)
        R = np.stack([generator.R(x) for i, x in enumerate(X)], axis=0)
        H_preds = {
        }
        R_preds = {
        }
        for run_name, model in models.items():
            X = torch.tensor(X).float()
            H_pred = model.model.grad_H(X)
            R_pred = model.model.reparam(X)[1]
            H_preds[run_name] = H_pred.detach().numpy()
            R_preds[run_name] = R_pred.detach().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    for run_name, H_pred in H_preds.items():
        idx = 3
        ax.scatter(grad_H[:, idx], H_pred[:, idx], label=run_name, color=colors[run_name])
    ax.set_ylabel(r"Identified $\frac{\partial H(x)}{\partial x_4}$", fontsize=64)
    ax.set_xlabel(r"True $\frac{\partial H(x)}{\partial x_4}$", fontsize=64)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.legend(fontsize=32)
    plt.savefig(os.path.join("results", "compare_H.png"))

    fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    for run_name, R_pred in R_preds.items():
        idx1 = 1
        idx2 = 1
        ax.scatter(R[:, idx1, idx2], R_pred[:, idx1, idx2], label=run_name, color=colors[run_name])
    ax.set_ylabel(r"Identified $R_{22}(x)$", fontsize=64)
    ax.set_xlabel(r"True $R_{22}(x)$", fontsize=64)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.legend(fontsize=32)
    plt.savefig(os.path.join("results", "compare_R.png"))



#noise_plots()
plot_scaling()
recipe()
compare()
prior_vs_default()

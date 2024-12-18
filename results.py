import mlflow
import numpy as np
from matplotlib import pyplot as plt
import os

#plt.style.use('seaborn-darkgrid')  # You can choose other styles like 'ggplot', 'classic', etc.

if not os.path.exists("results"):
    os.makedirs("results")

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
        ax[j].legend()
    plt.show()

    plt.savefig(os.path.join("results", "noise.png"))


def plot_scaling():
    experiment = mlflow.get_experiment_by_name("scaling")
    all_runs = mlflow.search_runs(experiment.experiment_id)
    for name in ["ball", "motor", "spring"]:
        runs = all_runs[all_runs["tags.name"] == name]
        baseline = {}
        sigmoid = {}
        matmul = {}
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
            elif run_name.startswith(f"{name}_sigmoid"):
                sigmoid[int(params["num_trajectories"])] = [mae_rel, mse_rel]
            elif run_name.startswith(f"{name}_matmul"):
                matmul[int(params["num_trajectories"])] = [mae_rel, mse_rel]
            elif run_name.startswith(f"{name}_prior"):
                prior[int(params["num_trajectories"])] = [mae_rel, mse_rel]
        assert matmul.keys() == baseline.keys() == sigmoid.keys() == prior.keys()
        sorted_keys = sorted(matmul.keys())
        mae_rel_baseline = [baseline[k][0] for k in sorted_keys]
        mae_rel_matmul = [matmul[k][0] for k in sorted_keys]
        mae_rel_prior = [prior[k][0] for k in sorted_keys]


        # Create the figure and axis with an optimized size
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted from (30, 30) to (12, 8)

        # Plotting the data with markers and increased line width for better visibility
        ax.plot(sorted_keys, mae_rel_baseline, label="Baseline", marker='o', linewidth=2)
        ax.plot(sorted_keys, mae_rel_matmul, label="Matmul", marker='s', linewidth=2)
        ax.plot(sorted_keys, mae_rel_prior, label="Prior", marker='^', linewidth=2)

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
        ax.legend(fontsize=14, loc='best')  # 'best' lets matplotlib decide the optimal location

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
    time = np.arange(len(X_true[idx1, :10000, state1])) * 0.01
    for j, (idx, state) in enumerate([(idx1, state1), (idx2, state2)]):
        ax[j].plot(time, X_true[idx, :10000, state], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j].plot(time, X_pred[idx, :10000, state], label=run_name)
        ax[j].set_xlabel("Time [s]")
        ax[j].set_ylabel("State {}".format(state))
        ax[j].legend()
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

    idx1 = 12
    state1 = 1
    idx2 = 11
    state2 = 0
    n = X_pred.shape[-1]
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    for j, (idx, state) in enumerate([(idx1, state1), (idx2, state2)]):
        ax[j].set_xlabel("Time [s]")
        ax[j].set_ylabel("State {}".format(state))
        t = np.arange(5000) * 0.01
        ax[j].plot(t, X_true[idx, :5000, state], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j].plot(t, X_pred[idx, :5000, state], label=run_name)
        ax[j].legend()
    # plt.show()
    plt.savefig(os.path.join("results", "compare.png"))



#noise_plots()
plot_scaling()
recipe()
compare()
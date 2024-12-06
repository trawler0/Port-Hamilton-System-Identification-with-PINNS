import mlflow
import numpy as np
from matplotlib import pyplot as plt
import os

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
            ax[j].plot(t, X_pred[0, :1000, j], label=f"dB={dB}")
        ax[j].plot(t, X_true[0, :1000, j], label="True", color="black")
        ax[j].set_xlabel("Time [s]")
        ax[j].set_ylabel("Trajectory")
        ax[j].grid()
        ax[j].set_title(f"Trajectory {j}")
        ax[j].legend()
    #plt.show()

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
        mse_rel_baseline = [baseline[k][1] for k in sorted_keys]
        mae_rel_sigmoid = [sigmoid[k][0] for k in sorted_keys]
        mse_rel_sigmoid = [sigmoid[k][1] for k in sorted_keys]
        mae_rel_matmul = [matmul[k][0] for k in sorted_keys]
        mse_rel_matmul = [matmul[k][1] for k in sorted_keys]
        mae_rel_prior = [prior[k][0] for k in sorted_keys]
        mse_rel_prior = [prior[k][1] for k in sorted_keys]

        fig, ax = plt.subplots(1, 2, figsize=(30, 15))
        ax[0].plot(sorted_keys, mae_rel_baseline, label="Baseline")
        ax[0].plot(sorted_keys, mae_rel_sigmoid, label="Sigmoid")
        ax[0].plot(sorted_keys, mae_rel_matmul, label="Matmul")
        ax[0].plot(sorted_keys, mae_rel_prior, label="Prior")
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_title(f"{name.capitalize()} MAE")
        ax[0].set_xlabel("Number of trajectories used for training")
        ax[0].set_ylabel("normalized MAE")
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(sorted_keys, mse_rel_baseline, label="Baseline")
        ax[1].plot(sorted_keys, mse_rel_sigmoid, label="Sigmoid")
        ax[1].plot(sorted_keys, mse_rel_matmul, label="Matmul")
        ax[1].plot(sorted_keys, mse_rel_prior, label="Prior")
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_title(f"{name.capitalize()} MSE")
        ax[1].set_xlabel("Number of trajectories used for training")
        ax[1].set_ylabel("normalized MSE")
        ax[1].grid()
        ax[1].legend()
        # plt.show()


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
    idx2 = 9
    idx3 = 7
    fig, ax = plt.subplots(n, 3, figsize=(30, 30))

    for j in range(n):
        ax[j, 0].plot(X_true[idx1, :10000, j], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j, 0].plot(X_pred[idx1, :10000, j], label=run_name)
        ax[j, 0].legend()
        ax[j, 1].plot(X_true[idx2, :10000, j], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j, 1].plot(X_pred[idx2, :10000, j], label=run_name)
        ax[j, 1].legend()
        ax[j, 2].plot(X_true[idx3, :10000, j], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j, 2].plot(X_pred[idx3, :10000, j], label=run_name)
        ax[j, 2].legend()
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

    idx1 = 8
    idx2 = 12
    idx3 = 11
    n = X_pred.shape[-1]
    fig, ax = plt.subplots(n, 3, figsize=(30, 30))

    for j in range(n):
        for i in range(3):
            ax[j, i].set_xlabel("Time [s]")
            ax[j, i].set_ylabel("Trajectory")
            ax[j, i].grid()
        t = np.arange(5000) * 0.01
        ax[j, 0].plot(t, X_true[idx1, :5000, j], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j, 0].plot(t, X_pred[idx1, :5000, j], label=run_name)
        ax[j, 0].legend()
        ax[j, 1].plot(t, X_true[idx2, :5000, j], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j, 1].plot(t, X_pred[idx2, :5000, j], label=run_name)
        ax[j, 1].legend()
        ax[j, 2].plot(t, X_true[idx3, :5000, j], label="True", color="black")
        for run_name, X_pred in preds.items():
            ax[j, 2].plot(t, X_pred[idx3, :5000, j], label=run_name)

        ax[j, 2].legend()
    # plt.show()
    plt.savefig(os.path.join("results", "compare.png"))



noise_plots()
plot_scaling()
recipe()
compare()
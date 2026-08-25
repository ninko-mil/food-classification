from pathlib import Path

import mlflow
import matplotlib.pyplot as plt


EXPERIMENT_NAME = "Food11 Final Model"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow eksperiment '{EXPERIMENT_NAME}' nije pronađen."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise RuntimeError("Nema MLflow run-ova.")

    run_id = runs.iloc[0]["run_id"]

    client = mlflow.tracking.MlflowClient()

    train_loss = client.get_metric_history(
        run_id,
        "train_loss"
    )

    train_accuracy = client.get_metric_history(
        run_id,
        "train_accuracy"
    )

    loss_steps = [m.step for m in train_loss]
    loss_values = [m.value for m in train_loss]

    acc_steps = [m.step for m in train_accuracy]
    acc_values = [m.value for m in train_accuracy]

    plt.figure(figsize=(8, 5))
    plt.plot(loss_steps, loss_values, marker="o")
    plt.title("Learning Curve - Training Loss")
    plt.xlabel("Epoha")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "learning_curve_loss.png",
        dpi=150
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(acc_steps, acc_values, marker="o")
    plt.title("Learning Curve - Training Accuracy")
    plt.xlabel("Epoha")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "learning_curve_accuracy.png",
        dpi=150
    )
    plt.close()

    print("Sačuvano:")
    print("results/learning_curve_loss.png")
    print("results/learning_curve_accuracy.png")


if __name__ == "__main__":
    main()
